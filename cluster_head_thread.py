import threading
import time
import numpy as np
import queue

class ClusterHeadThread(threading.Thread):
    def __init__(
        self,
        cluster_id,
        client_ids,
        phi,
        cluster_queue,
        server_queue,
        model_queue,
        broadcast_queues,
        timeout=120
    ):
        super().__init__()
        self.cluster_id = cluster_id
        self.client_ids = client_ids
        self.phi = phi
        self.cluster_queue = cluster_queue
        self.server_queue = server_queue
        self.broadcast_queues = broadcast_queues
        self.model_queue = model_queue
        self.timeout = timeout
        self.stop_signal = False
        self.round_counter = 0

    def run(self):
        print(f"[Cluster {self.cluster_id}] 🟢 Thread started with clients: {self.client_ids}")

        while not self.stop_signal:
            buffer = []
            start_time = time.time()
            min_required = max(1, int(len(self.client_ids) * self.phi))

            print(f"[Cluster {self.cluster_id}] ⏳ Waiting for {min_required} updates (have: {len(buffer)})")

            # Collect updates from participating clients
            while len(buffer) < min_required:
                try:
                    update = self.cluster_queue.get(timeout=60)
                    if update['cid'] in self.client_ids:
                        buffer.append(update)
                        print(f"[Cluster {self.cluster_id}] ✅ Received update from Client {update['cid']}")
                except queue.Empty:
                    if time.time() - start_time > self.timeout:
                        print(f"[ERROR][Cluster {self.cluster_id}] ⌛ Timeout while waiting for clients")
                        break

            if not buffer:
                print(f"[Cluster {self.cluster_id}] ❌ No client updates received — skipping round {self.round_counter}")
                self.round_counter += 1
                continue
            # for update in buffer:
                # print(f"[Cluster {self.cluster_id}] 📉 Client {update['cid']} staleness: {update['staleness']}")

            print(f"[Cluster {self.cluster_id}] 📦 Collected {len(buffer)} updates, starting aggregation")

            aggregated_model = self.aggregate_models([b["parameters"] for b in buffer])
            client_grads = {b["cid"]: b["gradient"] for b in buffer}

            self.server_queue.put({
                "cluster_id": self.cluster_id,
                "parameters": aggregated_model,
                "client_gradients": client_grads
            })
            # print(f"[Cluster {self.cluster_id}] 📤 Sent aggregated model to server (Round {self.round_counter})")

            # Send updated global model back to clients
            try:
                selected_clients = [update["cid"] for update in buffer]

            # Receive global model
                msg = self.model_queue.get(timeout=60)
                # print('ho',global_model)
                if msg == "STOP":
                    print(f"[Cluster {self.cluster_id}] 🛑 Received STOP from server")
                    break
                global_model, round_ = msg
                print(f"[Cluster {self.cluster_id}] 📥 Received global model from server")

                for cid in selected_clients:
                    self.broadcast_queues[cid].put((global_model, round_))
                    print(f"[Cluster {self.cluster_id}] 📬 Sent global model to Client {cid}")

            except queue.Empty:
                print(f"[ERROR][Cluster {self.cluster_id}] ❌ Server did not respond with global model")

            self.round_counter += 1

    def aggregate_models(self, model_list):
        aggregated = []
        for params in zip(*model_list):
            stacked = np.stack(params)
            aggregated.append(np.mean(stacked, axis=0))
        return aggregated

    def stop(self):
        self.stop_signal = True