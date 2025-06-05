import matplotlib.pyplot as plt
import ast # safer than eval() for parsing string representations of Python literals

# --- Configuration ---
file_name = "training.txt"
plot_output_name = "loss_and_lr_plot.png" # Changed output file name

# --- Data storage for plotting ---
training_epochs = []
training_losses = []
eval_epochs = []
eval_losses = []
lr_epochs = [] # New list for learning rate epochs
learning_rates = [] # New list for learning rates

# --- 1. Read data from training.txt and extract values for plotting ---
print(f"Attempting to read data from '{file_name}'...")
try:
    with open(file_name, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip() # Remove leading/trailing whitespace

            # Ignore lines that don't start with '{' (e.g., empty lines, incomplete dicts)
            if not line.startswith('{'):
                # print(f"Skipping line {line_num} (does not start with '{{'): '{line}'") # Uncomment for debugging skipped lines
                continue

            try:
                # Use ast.literal_eval to safely parse dictionary strings
                data = ast.literal_eval(line)

                # Extract Training Loss
                if 'loss' in data and 'eval_loss' not in data:
                    if 'epoch' in data and 'loss' in data:
                        training_epochs.append(data['epoch'])
                        training_losses.append(data['loss'])
                    else:
                        print(f"Warning: Line {line_num}: Training entry missing 'epoch' or 'loss'. Skipping: '{line}'")
                
                # Extract Evaluation Loss
                elif 'eval_loss' in data:
                    if 'epoch' in data and 'eval_loss' in data:
                        eval_epochs.append(data['epoch'])
                        eval_losses.append(data['eval_loss'])
                    else:
                        print(f"Warning: Line {line_num}: Evaluation entry missing 'epoch' or 'eval_loss'. Skipping: '{line}'")
                
                # Extract Learning Rate (present in many 'loss' entries)
                if 'learning_rate' in data and 'epoch' in data:
                    lr_epochs.append(data['epoch'])
                    learning_rates.append(data['learning_rate'])
                # Note: Learning rate might not be in every single log line,
                # but will be in most of the training loss lines.

            except (ValueError, SyntaxError) as e:
                print(f"Warning: Line {line_num}: Could not parse as dictionary. Skipping line: '{line}' (Error: {e})")
except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found. Please ensure it exists in the same directory as the script, or provide its full path.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the file: {e}")
    exit()

# --- 2. Plot the data ---
if not training_epochs and not eval_epochs and not lr_epochs:
    print("No valid training loss, evaluation loss, or learning rate data found to plot from the file.")
else:
    fig, ax1 = plt.subplots(figsize=(12, 7)) # Create a figure and the first axes (for loss)

    # Initialize plot line variables to None
    line1, line2, line3 = None, None, None

    # Plot Training Loss
    if training_epochs:
        sorted_training_data = sorted(zip(training_epochs, training_losses))
        train_e, train_l = zip(*sorted_training_data)
        line1, = ax1.plot(train_e, train_l, label='Training Loss', color='blue', alpha=0.8)

    # Plot Evaluation Loss
    if eval_epochs:
        sorted_eval_data = sorted(zip(eval_epochs, eval_losses))
        eval_e, eval_l = zip(*sorted_eval_data)
        line2, = ax1.plot(eval_e, eval_l, label='Validation Loss', color='red', marker='o', linestyle='--', markersize=5)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='black') # Label for the left Y-axis
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_title('LLM Training Progress: Loss and Learning Rate Over Epochs')
    ax1.grid(True) # Add a grid for easier reading of values


    # Create a second Y-axis for Learning Rate
    ax2 = ax1.twinx() 

    # Plot Learning Rate
    if learning_rates:
        sorted_lr_data = sorted(zip(lr_epochs, learning_rates))
        lr_e, lrs = zip(*sorted_lr_data)
        line3, = ax2.plot(lr_e, lrs, label='Learning Rate', color='green', linestyle=':', linewidth=2) # Dotted line for LR

    ax2.set_ylabel('Learning Rate', color='green') # Label for the right Y-axis
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Combine legends from both axes
    # We collect all artists (lines) and their labels
    # Filter out None values in case some plots didn't occur (e.g., no eval_loss data)
    lines = [line for line in [line1, line2, line3] if line is not None] 
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='best') # Place legend in the best available location

    plt.tight_layout() # Adjust plot to ensure everything fits

    # --- 3. Save the plot as a PNG file ---
    try:
        plt.savefig(plot_output_name)
        print(f"Plot saved successfully as '{plot_output_name}'.")
    except Exception as e:
        print(f"Error saving plot '{plot_output_name}': {e}")

    # Optional: If you want to display the plot immediately after saving
    # plt.show()