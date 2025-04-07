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
            # print(buffer[0]['s_it'][0].shape, buffer[0]['s_it'][-1].shape)
            aggregated_model = self.aggregate_models_ota([b["s_it"] for b in buffer], [update['h_it'] for update in buffer], update['lambda_t'])
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
    
    def aggregate_models_ota(self, s_it_list, h_list, lambda_t, noise_std=0.01):
        aggregated_model = []
        num_clients = len(s_it_list)
        num_layers = len(s_it_list[0])

        for layer_idx in range(num_layers):
            sum_signal = np.zeros_like(s_it_list[0][layer_idx], dtype=np.complex128)

            for client_idx in range(num_clients):
                s = s_it_list[client_idx][layer_idx]
                h = h_list[client_idx][layer_idx]

                # Debugging
                if s.shape != h.shape:
                    print(f"[ERROR] Mismatch in shapes for client {client_idx}, layer {layer_idx}: s {s.shape}, h {h.shape}")
                    continue

                sum_signal += h * s

            # Add complex Gaussian noise
            noise = self.generate_awgn_for_layer(sum_signal.shape, std=noise_std)
            noisy_signal = sum_signal + noise

            # Final OTA output (real part scaled by lambda)
            aggregated_model.append(np.real(noisy_signal * lambda_t))

        return aggregated_model

    def generate_awgn_for_layer(self, shape, std=0.01):
        real_noise = np.random.normal(loc=0.0, scale=std, size=shape)
        imag_noise = np.random.normal(loc=0.0, scale=std, size=shape)
        return real_noise + 1j * imag_noise

    def stop(self):
        self.stop_signal = True