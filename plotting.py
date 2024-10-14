import matplotlib.pyplot as plt


def save_accuracies_plot(training_accuracies, testing_accuracies):
    num_models = len(training_accuracies)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Training
    axs[0].set_title(f"Training Accuracy for {num_models} Networks", fontsize=16)
    axs[0].set_xlabel('Epochs', fontsize=14)
    axs[0].set_ylabel('Accuracy', fontsize=14)
    axs[0].grid()

    for model, accuracy in training_accuracies.items():
        axs[0].plot(accuracy, label=f"{model} - Train")
    
    axs[0].legend()
    
    # Testing 
    axs[1].set_title(f"Dev Accuracy for {num_models} Networks", fontsize=16)
    axs[1].set_xlabel('Epochs', fontsize=14)
    axs[1].set_ylabel('Accuracy', fontsize=14)
    axs[1].grid()

    for model, accuracy in testing_accuracies.items():
        axs[1].plot(accuracy, label=f"{model} - Dev")
    
    axs[1].legend()

    accuracy_file = f'./images/accuracy_{num_models}_networks.png'
    plt.tight_layout()
    plt.savefig(accuracy_file)
    plt.close()
    print(f"\n\nAccuracy plot saved as {accuracy_file}")

def prev(nn2_train_accuracy, nn3_train_accuracy, nn2_test_accuracy, nn3_test_accuracy):
    # Plot the training accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(nn2_train_accuracy, label='2 layers')
    plt.plot(nn3_train_accuracy, label='3 layers')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy for 2, 3 Layer Networks')
    plt.legend()
    plt.grid()

    # Save the training accuracy figure
    training_accuracy_file = 'train_accuracy.png'
    plt.savefig(training_accuracy_file)
    print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

    # Plot the testing accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(nn2_test_accuracy, label='2 layers')
    plt.plot(nn3_test_accuracy, label='3 layers')
    plt.xlabel('Epochs')
    plt.ylabel('Dev Accuracy')
    plt.title('Dev Accuracy for 2 and 3 Layer Networks')
    plt.legend()
    plt.grid()

    # Save the testing accuracy figure
    testing_accuracy_file = 'dev_accuracy.png'
    plt.savefig(testing_accuracy_file)
    print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")
    # plt.show()

