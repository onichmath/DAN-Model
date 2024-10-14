import matplotlib.pyplot as plt


def save_accuracies_plot(training_accuracies, testing_accuracies):
    # Plot the training accuracy
    plt.figure(figsize=(8, 6))
    for model, accuracy in training_accuracies.items():
        plt.plot(accuracy, label=f"{model} - Train")
    for model, accuracy in testing_accuracies.items():
        plt.plot(accuracy, label=f"{model} - Dev")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f"Accuracy for {len(training_accuracies)} Networks")
    plt.legend()
    plt.grid()

    # Save the training accuracy figure
    accuracy_file = 'accuracy.png'
    plt.savefig(accuracy_file)
    print(f"\n\nAccuracy plot saved as {accuracy_file}")

    # plt.show()


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

