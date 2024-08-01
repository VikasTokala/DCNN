import matplotlib.pyplot as plt
import numpy as np
import librosa

def plot_averaged_magnitude(encoder_output_magnitude,title='Encoder Output - Layer 1',clabel='Magnitude[dB]',
                            fig_name='Encoder1.pdf',xlab='Frequency dimension', ylab='Time dimension'):
    """
    Plots the averaged magnitude of the complex-valued encoder output.
    
    Parameters:
    encoder_output_magnitude (numpy.ndarray): The magnitude of the encoder output with shape [batch_size, channels, height, width].
    """
    # Select the first example from the batch
    # batch_index = 0
    # magnitude = encoder_output_magnitude[batch_index]

    # Compute the average magnitude across all feature maps
    average_magnitude =encoder_output_magnitude

    # Normalize the averaged magnitude for better visualization
    # normalized_magnitude = (average_magnitude - average_magnitude.min()) / (average_magnitude.max() - average_magnitude.min())
    
    # Plot the averaged magnitude
    plt.figure(figsize=(6, 5))
    plt.imshow((average_magnitude), aspect='auto', cmap='magma',origin='lower')
    plt.title(title,fontname="Times New Roman",fontweight='bold',fontsize='18')
    plt.colorbar(label=clabel)
    plt.xlabel(xlab,fontname="Times New Roman",fontweight='bold',fontsize='16')
    plt.ylabel(ylab,fontname="Times New Roman",fontweight='bold',fontsize='16')
    # plt.show()
    plt.savefig(fig_name, format="pdf", bbox_inches="tight")

# Example usage
# # Replace this with the actual magnitude output from your encoder
# output = np.random.rand(32, 128, 4, 327) + 1j * np.random.rand(32, 128, 4, 327)  # Complex-valued output
# magnitude_output = np.abs(output)  # Compute the magnitude

# Call the function with the magnitude output
# plot_averaged_magnitude(magnitude_output)