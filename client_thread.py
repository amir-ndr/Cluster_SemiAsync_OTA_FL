import threading
import time
import queue
import torch
import torch.nn.functional as F
from model import CNNMnist
from torch.utils.data import DataLoader

class ClientThread(threading.Thread):
    def __init__(self, cid, model, train_dataset, test_dataset, cluster_queue, receive_model_queue, device, batch_size=512):
        super().__init__()
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64)
        self.cluster_queue = cluster_queue
        self.receive_model_queue = receive_model_queue
        self.device = device
        self.stop_signal = threading.Event()  # Changed to Event for better control
        self.round_counter = 0
        self.local_staleness = 0
        self.last_participated_round = -1
        self.staleness_log = []
        self.participation_count = 0
        self.skipped_rounds = 0
        self.last_activity = time.time()  # For deadlock detection

    def run(self):
        last_model = None  # Cache last model received

        while not self.stop_signal.is_set():
            try:
                try:
                    msg = self.receive_model_queue.get(timeout=30)
                    if msg == "STOP":
                        print(f"[Client {self.cid}] 🛑 Received STOP")
                        break

                    # A new model is received
                    print('hiiii',msg['round'])
                    last_model = msg['model']
                    staleness = self.local_staleness  # ← use accumulated staleness
                    self.local_staleness = 0          # ← reset after using it

                except queue.Empty:
                    print(f"[Client {self.cid}] ❗ No new model this round — using stale model")
                    self.local_staleness += 1         # ← stale, so increment
                    staleness = self.local_staleness
                    if last_model is None:
                        print(f"[Client {self.cid}] 🚫 No model ever received — skipping")
                        continue

                self.set_parameters(last_model)
                self.participation_count += 1

                print(f"[Client {self.cid}] Starting training (Round {self.round_counter + 1}) | 🕒 Staleness: {staleness}")


                # 3. Training
                train_start = time.time()
                optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
                self.model.train()
                self.model.to(self.device)

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
                    "parameters": self.get_parameters(),
                    "gradient": gradient,
                    "train_time": train_time,
                    "test_accuracy": test_acc,
                    "staleness": staleness,
                    "participation_count": self.participation_count
                }

                if self.cluster_queue:
                    self.cluster_queue.put(update, timeout=30)
                    print(f"[Client {self.cid}] ✅ Sent update | Acc: {test_acc:.2f} | Staleness: {staleness}")
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

    def set_parameters(self, parameters):
        if self.model is None:
            self.model = CNNMnist().to(self.device)  # Create model only when first update arrives
        if isinstance(parameters, dict) and 'params' in parameters:
            parameters = parameters['params']
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