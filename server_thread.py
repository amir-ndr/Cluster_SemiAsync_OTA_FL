import threading
import torch
from model import CNNMnist
import queue
import numpy as np
from sklearn.cluster import KMeans

class ServerThread(threading.Thread):
    def __init__(
        self,
        num_clusters,
        server_queue,
        global_model_queues,
        test_dataset,
        device, 
        num_rounds=15,
        recluster_interval=5
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.server_queue = server_queue  # cluster heads push here
        self.global_model_queues = global_model_queues  # cluster_id -> [Queue to send model to CH]
        self.test_dataset = test_dataset
        self.device = device
        self.num_rounds = num_rounds
        self.recluster_interval = recluster_interval
        self.round_counter = 0
        self.global_round = 0
        self.stop_signal = False
        self.client_gradients = {}  # client_id -> last gradient
        self.perform_reclustering_callback = None

    def run(self):
        while not self.stop_signal and self.round_counter < self.num_rounds:
            print(f"\n\n[Server] Waiting for updates from {self.num_clusters} cluster heads (Round {self.round_counter + 1})...\n")
            round_updates = []
            received_clusters = set()

            while len(received_clusters) < self.num_clusters:
                try:
                    update = self.server_queue.get(timeout=60)
                    cluster_id = update["cluster_id"]
                    if cluster_id not in received_clusters:
                        round_updates.append(update["parameters"])
                        received_clusters.add(cluster_id)
                        # print(f"[Server] Received update from Cluster {cluster_id}")
                        print(f"[Server] Received update from Cluster {cluster_id} (Clients: {list(update['client_gradients'].keys())})")

                        for client_id, grad in update.get("client_gradients", {}).items():
                            self.client_gradients[client_id] = grad

                except queue.Empty:
                    print("[Server] Timeout waiting for all clusters")
                    break

            if not round_updates:
                print("[Server] No updates to aggregate")
                continue

            print(f"[Server] Queue size: {self.server_queue.qsize()}")

            global_model = self.aggregate(round_updates)
            accuracy = self.evaluate_global_model(global_model)
            print(f"[Server] 🧪 Global Model Accuracy after Round {self.round_counter + 1}: {accuracy:.4f}")
            print("[Server] Global model aggregated and broadcasting to clusters")

            for cluster_id in range(self.num_clusters):
                # global_model = {
                #     'params': self.aggregate(round_updates),
                #     'round': self.round_counter
                #         }
                self.global_model_queues[cluster_id].put((global_model, self.global_round))

            if self.should_recluster():
                print(f"[Server] Re-clustering triggered at round {self.round_counter + 1}")
                self.perform_reclustering()

            self.round_counter += 1
            if self.round_counter >= self.num_rounds:
                print("[Server] ✅ Finished all training rounds — sending stop signal")
                for q in self.global_model_queues.values():
                    q.put("STOP")

    def aggregate(self, model_list):
        self.global_round+=1
        aggregated = []
        for params in zip(*model_list):
            stacked = np.stack(params)
            aggregated.append(np.mean(stacked, axis=0))
        return aggregated

    def evaluate_global_model(self, global_model_params):
        model = CNNMnist().to(self.device)
        model.eval()

        # Load global parameters into model
        for param, new_val in zip(model.parameters(), global_model_params):
            param.data = torch.tensor(new_val, dtype=param.dtype, device=self.device)

        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=64)
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                preds = output.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        return acc


    def should_recluster(self):
        return self.round_counter % self.recluster_interval == 0 and self.round_counter > 0

    def perform_reclustering(self):
        print("[Server] Performing re-clustering using KMeans on gradient similarity")

        if not self.client_gradients:
            print("[Server] No gradients to recluster with.")
            return

        client_ids = list(self.client_gradients.keys())
        gradient_matrix = np.stack([self.client_gradients[cid] for cid in client_ids])

        norms = np.linalg.norm(gradient_matrix, axis=1, keepdims=True)
        gradient_matrix_normalized = gradient_matrix / (norms + 1e-8)

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        labels = kmeans.fit_predict(gradient_matrix_normalized)

        client_cluster_map = {cid: int(cluster_id) for cid, cluster_id in zip(client_ids, labels)}

        print(f"\n[Server] 🔄 Re-clustering result at Round {self.round_counter}:")
        for cluster_id in range(self.num_clusters):
            members = [cid for cid, c in client_cluster_map.items() if c == cluster_id]
            print(f"  - Cluster {cluster_id}: Clients {members}")

        if self.perform_reclustering_callback:
            self.perform_reclustering_callback(client_cluster_map)

    def stop(self):
        self.stop_signal = True