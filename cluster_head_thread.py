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
        self.power_factor = 1.0

    def initialize_channel_state(self, model):
        h_nt = []
        for layer in model:
            shape = np.array(layer).shape
            real_part = np.random.normal(0, 1, size=shape)
            imag_part = np.random.normal(0, 1, size=shape)
            complex_channel = real_part + 1j * imag_part
            h_nt.append(complex_channel)
        return h_nt

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
                        D_i = update['num_samples']
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

            print(f"[Cluster {self.cluster_id}] 📦 Collected {len(buffer)} updates, starting aggregation")

            num_samples = sum(update["num_samples"] for update in buffer)
            aggregated_model = self.aggregate_models_ota([b["s_it"] for b in buffer],
                 [update['h_it'] for update in buffer], update['lambda_t'], [update['num_samples']/num_samples for update in buffer])
            self.ch_state = self.initialize_channel_state(aggregated_model)
            transmit_recover_model = self.transmit_ota(aggregated_model)
            client_grads = {b["cid"]: b["gradient"] for b in buffer}

            self.server_queue.put({
                "cluster_id": self.cluster_id,
                # "parameters": aggregated_model,
                "s_nt": transmit_recover_model,
                "h_nt": self.ch_state,
                "mu_t": self.power_factor,
                "client_gradients": client_grads,
                "mu_t": self.power_factor,
                "num_samples": num_samples
            })

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
    
    def aggregate_models_ota(self, s_it_list, h_list, lambda_t, weight, noise_std=0.01):
        aggregated_model = []
        num_clients = len(s_it_list)
        num_layers = len(s_it_list[0])

        for layer_idx in range(num_layers):
            sum_signal = np.zeros_like(s_it_list[0][layer_idx], dtype=np.complex128)

            for client_idx in range(num_clients):
                s = s_it_list[client_idx][layer_idx] * weight[client_idx]
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

    def generate_channel_inversion_vector(self):
        c_nt = []
        for h_layer in self.ch_state:
            h_layer = np.asarray(h_layer, dtype=np.complex128)
            magnitude_squared = np.abs(h_layer) ** 2 + 1e-8 
            c_layer = h_layer / magnitude_squared
            c_nt.append(c_layer)

        return c_nt

    def transmit_ota(self, aggregated_model):
        c_nt = self.generate_channel_inversion_vector()
        s_nt = [
            (1 / self.power_factor) * c * x 
            for c, x in zip(c_nt, aggregated_model)
        ]
        return s_nt

    def generate_awgn_for_layer(self, shape, std=0.01):
        real_noise = np.random.normal(loc=0.0, scale=std, size=shape)
        imag_noise = np.random.normal(loc=0.0, scale=std, size=shape)
        return real_noise + 1j * imag_noise

    def stop(self):
        self.stop_signal = True