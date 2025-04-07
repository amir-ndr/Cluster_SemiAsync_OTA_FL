import torch
import queue
from model import CNNMnist
from data_loader import load_mnist, partition_mnist_noniid
from client_thread import ClientThread
from cluster_head_thread import ClusterHeadThread
from server_thread import ServerThread

NUM_CLIENTS = 10
NUM_CLUSTERS = 2
PHI = 1
NUM_ROUNDS = 20
INTERVAL_RECLUSTER = 5

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load and partition dataset
train_dataset, test_dataset = load_mnist()
client_data_map = partition_mnist_noniid(train_dataset, num_clients=NUM_CLIENTS, shards_per_client=2)
for i in range(NUM_CLIENTS):
    print(f"[Client {i}] has {len(client_data_map[i])} samples | first 5 indices: {client_data_map[i][:5]}")


# 2. Setup communication channels
cluster_queues = {cid: queue.Queue() for cid in range(NUM_CLUSTERS)} #client to ch
server_queue = queue.Queue() # ch to server
client_model_queues = {i: queue.Queue() for i in range(NUM_CLIENTS)}  # send model to client
ch_model_queues = {cid: queue.Queue() for cid in range(NUM_CLUSTERS)}  # send model to CH

# 3. Create clients
clients = []

for i in range(NUM_CLIENTS):
    train_subset = torch.utils.data.Subset(train_dataset, client_data_map[i])
    model = CNNMnist()
    client = ClientThread(
        cid=i,
        model=model,
        train_dataset=train_subset,
        test_dataset=test_dataset,
        cluster_queue=None,  # will be set by cluster assignment
        receive_model_queue=client_model_queues[i],
        device=device,
        batch_size=512
    )
    clients.append(client)

# 4. Define cluster head launcher with dynamic mapping
cluster_heads = []
cluster_assignments = {}  # cid -> cluster_id

def launch_cluster_heads(client_cluster_map):
    print("\n[Main] Initializing cluster heads...")
    global cluster_heads, cluster_queues

    for cid, q in cluster_queues.items():
        if not isinstance(q, queue.Queue):
            raise ValueError(f"Cluster {cid} queue is invalid")

    # Stop existing cluster heads if any
    for q in cluster_queues.values():
        with q.mutex:
            q.queue.clear()
    for ch in cluster_heads:
        ch.stop()
    cluster_heads = []

    # Build new mapping
    new_cluster_queues = {cid: queue.Queue() for cid in range(NUM_CLUSTERS)}
    cluster_members = {cid: [] for cid in range(NUM_CLUSTERS)}

    for cid, cluster_id in client_cluster_map.items():
        cluster_members[cluster_id].append(cid)
        clients[cid].cluster_queue = new_cluster_queues[cluster_id]
        cluster_assignments[cid] = cluster_id

    print("\n[Cluster Assignments]")
    for cluster_id in range(NUM_CLUSTERS):
        members = cluster_members[cluster_id]
        print(f"  Cluster {cluster_id}: Clients {sorted(members)}")

    # Launch new cluster head threads
    for cluster_id in range(NUM_CLUSTERS):
        ch = ClusterHeadThread(
            cluster_id=cluster_id,
            client_ids=cluster_members[cluster_id],
            phi=PHI,
            cluster_queue=new_cluster_queues[cluster_id],
            server_queue=server_queue,
            model_queue=ch_model_queues[cluster_id],
            broadcast_queues=client_model_queues,
        )
        ch.start()
        cluster_heads.append(ch)

    cluster_queues = new_cluster_queues
    print(f"[Main] Cluster heads launched with updated assignments: {cluster_assignments}")

#  Create and start server thread
server = ServerThread(
    num_clusters=NUM_CLUSTERS,
    server_queue=server_queue,
    global_model_queues=ch_model_queues,
    test_dataset=test_dataset,
    device=device,
    num_rounds=NUM_ROUNDS,
    recluster_interval= INTERVAL_RECLUSTER
)
server.perform_reclustering_callback = launch_cluster_heads

# 2. Initial cluster assignment
initial_assignments = {cid: cid % NUM_CLUSTERS for cid in range(NUM_CLIENTS)}
launch_cluster_heads(initial_assignments)  # This populates cluster_assignments

# 3. Connect clients to their cluster queues
for client in clients:
    cluster_id = initial_assignments[client.cid]
    client.cluster_queue = cluster_queues[cluster_id]

server.start()

# 4. Start all clients
for c in clients:
    c.start()

# 7. Initial broadcast to clients (dummy global model)
initial_model = model.state_dict()
initial_params = [param.detach().cpu().numpy() for param in model.parameters()]
for cid in range(NUM_CLIENTS):
    client_model_queues[cid].put((initial_params, 1))

# 8. Wait for server to complete
server.join()
print("\n[Main] Training complete. Stopping all threads.")

# 9. Graceful shutdown
for ch in cluster_heads:
    ch.stop()
for c in clients:
    c.stop()