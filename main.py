model.eval()
    test_correct = 0
    test_total = 0
    
    print("\n🔬 Evaluating Model Performance Against Unseen Test Set...")
    with torch.no_grad():
        for feat, label in test_loader:
            feat, label = feat.to(device), label.to(device)
            
            output = model(feat)
            _, predicted = output.max(1)
            
            test_total += label.size(0)
            test_correct += predicted.eq(label).sum().item()
            
    final_test_accuracy = (test_correct / test_total) * 100
    print(f"📊 Final Accuracy on the Test Set: {final_test_accuracy:.2f}%")

    #torch.save(model.state_dict(), "Pytorch_baseline_b4.pth")
    #print("\n🎉 B1 Scene Model saved successfully as 'Pytorch_baseline_b4.pth'!")