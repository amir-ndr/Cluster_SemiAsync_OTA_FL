import threading
import time
import queue
import torch
import torch.nn.functional as F
from model import CNNMnist
import numpy as np
from torch.utils.data import DataLoader

class ClientThread(threading.Thread):
    def __init__(self, cid, model, train_dataset, test_dataset, 
            cluster_queue, receive_model_queue, device, batch_size):
        super().__init__()
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64)
        self.cluster_queue = cluster_queue
        self.receive_model_queue = receive_model_queue
        self.device = device
        self.stop_signal = threading.Event()  # Changed to Event for better control
        self.round_counter = 0
        self.staleness_log = []
        self.participation_count = 0
        self.skipped_rounds = 0
        self.power_factor = 1.0
        self.ch_state = self.initialize_channel_state()
        self.stale_weight = 1.0
        self.last_activity = time.time()  # For deadlock detection

    def initialize_channel_state(self):
        h_it = []
        for param in self.model.parameters():
            shape = param.data.cpu().numpy().shape
            real_part = np.random.normal(0, 1, size=shape)
            imag_part = np.random.normal(0, 1, size=shape)
            complex_channel = real_part + 1j * imag_part
            h_it.append(complex_channel)
        return h_it


    def run(self):
        self.last_model = None
        self.last_participation_round = 0  # Last round this client participated in
        self.server_global_round = -1       # Tracks true server round (from received models)

        while not self.stop_signal.is_set():
            try:
                # Check for new model from cluster head
                try:
                    msg = self.receive_model_queue.get(timeout=30)
                    if msg == "STOP":
                        print(f"[Client {self.cid}] 🛑 Received STOP")
                        break
                    elif isinstance(msg, tuple) and msg[0] == "ROUND_UPDATE":
                        _, global_round = msg
                        self.server_global_round = max(self.server_global_round, global_round)
                        print(f"[Client {self.cid}] 🔄 Updated to Server Round {global_round}")
                    else:  # Legacy format fallback
                        global_model, global_round = msg

                    self.last_model = global_model
                    # self.server_global_round = global_round
                    self.server_global_round = max(self.server_global_round, global_round)

                    print(f"[Client {self.cid}] 📥 Received model for Global Round {global_round}")

                except queue.Empty:
                    print(f"[Client {self.cid}] ❗ Stale Model - using cached model")
                    if self.last_model is None:
                        print(f"[Client {self.cid}] 🚫 No model available - skipping")
                        continue

                # Calculate true staleness:
                # How many global rounds passed since last participation
                # if self.last_participation_round == 0:
                #     staleness = 0
                # else:
                staleness = max(0, self.server_global_round - self.last_participation_round)
                if staleness > 0:
                    self.stale_weght = 1/staleness
                else:
                    self.stale_weght = 1.0
                self.stale_weght = 1.0

                print(f"[Client {self.cid}] start training 🕒 Staleness: {staleness} "
                    f"(Server Round: {self.server_global_round}, "
                    f"Last Participated: {self.last_participation_round})")

                # Train with current model
                self.set_parameters(self.last_model)
                self.participation_count += 1
                # self.last_participation_round = self.server_global_round  # Update participation

                # 3. Training
                train_start = time.time()
                # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

                self.model.train()

                for x, y in self.train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x)
                    loss = F.cross_entropy(output, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                train_time = time.time() - train_start
                gradient = self.get_flattened_gradient()
                test_loss, test_acc = self.evaluate()

                # 4. Send update
                update = {
                    "cid": self.cid,
                    # "parameters": self.get_parameters(),
                    "s_it": self.get_parameters_ota(),
                    "h_it": self.ch_state,
                    "lambda_t": self.power_factor,
                    "gradient": gradient,
                    "train_time": train_time,
                    "test_accuracy": test_acc,
                    "num_samples": len(self.train_loader.dataset)
                    # "staleness": staleness,
                    # "participation_count": self.participation_count
                }

                if self.cluster_queue:
                    try:
                        self.cluster_queue.put(update, timeout=5)  # Shorter timeout
                        self.last_participation_round = self.server_global_round
                    except queue.Full:
                        print(f"[Client {self.cid}] ❌ Failed to participate this round")
                        staleness = max(0, self.server_global_round - self.last_participation_round)

                    print(f"[Client {self.cid}] ✅ Sent update | Loss: {test_loss:.2f} | Acc: {test_acc:.2f} | Staleness: {staleness} | Train time: {train_time}s")
                else:
                    print(f"[Client {self.cid}] ❌ ERROR: No cluster queue")

                self.round_counter += 1

            except Exception as e:
                print(f"[Client {self.cid}] ❌ CRITICAL ERROR: {e}")
                self.stop_signal.set()
                break

    def stop(self):
        self.stop_signal.set()

    def get_parameters(self):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]
    
    def get_parameters_ota(self):
        x_it = self.get_parameters()
        b_it = self.generate_channel_inversion_vector()
        s_it = [
            (1 / self.power_factor) * self.stale_weight * b * x 
            for b, x in zip(b_it, x_it)
        ]
        return s_it

    def generate_channel_inversion_vector(self):
        b_it = []
        for h_layer in self.ch_state:
            # Ensure complex type
            h_layer = np.asarray(h_layer, dtype=np.complex128)
            # Compute b = h / |h|^2 for each entry
            magnitude_squared = np.abs(h_layer) ** 2 + 1e-8 
            b_layer = h_layer / magnitude_squared

            b_it.append(b_layer)

        return b_it

    def set_parameters(self, parameters):
        if self.model is None:
            self.model = CNNMnist().to(self.device)  # Create model only when first update arrives
        for param, new_val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_val, dtype=param.dtype, device=self.device)

    def get_flattened_gradient(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1).detach().cpu())
            else:
                grads.append(torch.zeros_like(param).view(-1).cpu())
        return torch.cat(grads).numpy()

    def evaluate(self):
        self.model.eval()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss += F.cross_entropy(output, y, reduction='sum').item()
                correct += (output.argmax(dim=1) == y).sum().item()

        loss /= len(self.test_loader.dataset)
        acc = correct / len(self.test_loader.dataset)
        return loss, acc