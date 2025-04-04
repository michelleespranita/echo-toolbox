import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def play_video(video):
    """Plays a video.

    Args:
        video: A np.ndarray of shape (T, H, W, C)
    """
    if len(video.shape) == 3:
        print(f'Video cannot be played! Shape: {video.shape}')
        plt.imshow(video)
        return

    fig, ax = plt.subplots()
    im = ax.imshow(video[0], cmap="gray")  # Display first frame in grayscale

    # 🔄 Update function for animation
    def update(frame):
        im.set_array(video[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(video), interval=30)

    plt.close(fig)

    return HTML(ani.to_jshtml())