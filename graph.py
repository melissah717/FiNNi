import matplotlib.pyplot as plt

def plot_vs(val_df):
    plt.figure(figsize=(10, 6))
    plt.scatter(val_df['Actual'], val_df['Predicted'], alpha=0.7)
    plt.plot([val_df['Actual'].min(), val_df['Actual'].max()], 
             [val_df['Actual'].min(), val_df['Actual'].max()], 
             color='red', linestyle='--', label="Perfect Prediction")
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.grid(alpha=0.6)
    plt.show()
    
def plot_training_validation_loss(history):
    """
    Plot training and validation loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()