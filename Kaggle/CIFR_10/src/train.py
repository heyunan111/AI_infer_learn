import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
import os
from PIL import Image
from tqdm import tqdm


def train(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs=50,
    patience=10,
    min_delta=0.001,
):
    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for data in pbar:
            inputs = data["image"].to(device)
            labels = data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{100*train_correct/train_total:.2f}%",
                }
            )
            # 验证
        val_acc, val_loss = validate(model, device, test_loader, criterion)

        # 记录指标
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}: LR={current_lr:.6f}, Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%"
        )

        # 早停检查
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"💾 保存最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"⏰ 早停计数器: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"🛑 早停触发！最佳验证准确率: {best_val_acc:.2f}%")
            break

    return {
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }


def validate(net, device, test_loader, criterion):
    net.eval()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for data in test_loader:
            inputs = data["image"]
            labels = data["label"]

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(test_loader)
    print(
        f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}% ({correct}/{total})"
    )
    return accuracy, avg_val_loss


def determine_T_0(total_epochs, T_mult=1.5, num_restarts=3, min_T_0=10):
    """
    更稳健的T_0计算函数
    """
    # 参数验证
    if total_epochs < min_T_0 * 2:
        return max(min_T_0, total_epochs // 2)

    # 简单实用的策略：基于经验公式
    if T_mult == 1:
        # 固定周期
        T_0 = total_epochs // (num_restarts + 1)
    elif T_mult < 1.3:
        # 缓慢增长
        T_0 = total_epochs // (num_restarts + 2)
    elif T_mult < 1.8:
        # 中等增长
        T_0 = total_epochs // (num_restarts + 3)
    else:
        # 快速增长
        T_0 = total_epochs // (num_restarts + 4)

    # 应用约束
    T_0 = max(T_0, min_T_0)
    T_0 = min(T_0, total_epochs // 2)

    return T_0
