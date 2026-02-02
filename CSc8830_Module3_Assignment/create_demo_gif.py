#!/usr/bin/env python3#!/usr/bin/env python3#!/usr/bin/env python3

"""

Create a visually stunning GIF demonstration of the Convolution Theorem.""""""

Features dark theme, professional styling, and clear step-by-step explanations.

"""Create a visually stunning GIF demonstration of the Convolution Theorem.Create a GIF demonstration of the Convolution Theorem Proof.



import numpy as npFeatures smooth animations, professional styling, and clear explanations.Shows the step-by-step process of spatial vs frequency domain blurring.

import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch, Circle""""""

from PIL import Image

import os



# Dark theme colorsimport numpy as npimport numpy as np

COLORS = {

    'bg': '#0d1117',import matplotlib.pyplot as pltimport matplotlib.pyplot as plt

    'text': '#e6edf3',

    'accent': '#58a6ff',from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatchfrom PIL import Image

    'success': '#3fb950',

    'highlight': '#f85149',from matplotlib.colors import LinearSegmentedColormapimport os

    'purple': '#a371f7',

    'gold': '#d29922'from PIL import Image

}

import osdef create_sample_image(size=256):

plt.rcParams['figure.facecolor'] = COLORS['bg']

plt.rcParams['axes.facecolor'] = COLORS['bg']    """Create a sample test image."""

plt.rcParams['text.color'] = COLORS['text']

plt.rcParams['axes.labelcolor'] = COLORS['text']# Custom color scheme    image = np.zeros((size, size), dtype=np.float64)

plt.rcParams['xtick.color'] = COLORS['text']

plt.rcParams['ytick.color'] = COLORS['text']COLORS = {    



    'bg': '#0a0a0a',    # Checkerboard

def create_sample_image(size=256):

    """Create a visually interesting test image."""    'text': '#ffffff',    block = size // 8

    image = np.zeros((size, size), dtype=np.float64)

    y, x = np.ogrid[:size, :size]    'accent': '#00d4ff',    for i in range(8):

    center = size // 2

        'success': '#00ff88',        for j in range(8):

    # Concentric circles

    for r in range(20, center, 25):    'highlight': '#ff6b6b',            if (i + j) % 2 == 0:

        ring = ((x - center)**2 + (y - center)**2 >= (r-3)**2) & \

               ((x - center)**2 + (y - center)**2 <= (r+3)**2)    'purple': '#a855f7',                image[i*block:(i+1)*block, j*block:(j+1)*block] = 0.8

        image[ring] = 0.7

        'gold': '#fbbf24'    

    # Center bright spot

    mask = (x - center)**2 + (y - center)**2 <= 30**2}    # Circle

    image[mask] = 1.0

        y, x = np.ogrid[:size, :size]

    # Corner squares

    for cx, cy in [(50, 50), (50, 206), (206, 50), (206, 206)]:def create_sample_image(size=256):    center = size // 2

        image[cy-15:cy+15, cx-15:cx+15] = 0.9

        """Create a visually interesting test image."""    r = size // 6

    return image

    image = np.zeros((size, size), dtype=np.float64)    mask = (x - center)**2 + (y - center)**2 <= r**2



def create_gaussian_kernel(size=15, sigma=2.5):    y, x = np.ogrid[:size, :size]    image[mask] = 1.0

    """Create a normalized Gaussian kernel."""

    ax = np.arange(-size // 2 + 1, size // 2 + 1)    center = size // 2    

    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))        return image

    return kernel / kernel.sum()

    # Concentric circles



def spatial_blur(image, kernel):    for r in range(20, center, 25):def create_gaussian_kernel(size=15, sigma=2.5):

    """Blur using scipy spatial convolution."""

    from scipy.ndimage import convolve        ring = ((x - center)**2 + (y - center)**2 >= (r-3)**2) & \    """Create a normalized Gaussian kernel."""

    return convolve(image, kernel, mode='constant')

               ((x - center)**2 + (y - center)**2 <= (r+3)**2)    ax = np.arange(-size // 2 + 1, size // 2 + 1)



def frequency_blur(image, kernel):        image[ring] = 0.7    xx, yy = np.meshgrid(ax, ax)

    """Blur using FFT multiplication."""

    H, W = image.shape        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    kh, kw = kernel.shape

    ph, pw = H + kh - 1, W + kw - 1    # Center bright spot    return kernel / kernel.sum()

    

    image_padded = np.zeros((ph, pw), dtype=np.float64)    mask = (x - center)**2 + (y - center)**2 <= 30**2

    image_padded[:H, :W] = image

        image[mask] = 1.0def spatial_blur(image, kernel):

    kernel_padded = np.zeros((ph, pw), dtype=np.float64)

    kernel_padded[:kh, :kw] = kernel        """Blur using scipy spatial convolution."""

    

    F_image = np.fft.fft2(image_padded)    # Corner squares    from scipy.ndimage import convolve

    F_kernel = np.fft.fft2(kernel_padded)

    result = np.fft.ifft2(F_image * F_kernel).real    for cx, cy in [(50, 50), (50, 206), (206, 50), (206, 206)]:    return convolve(image, kernel, mode='constant')

    

    return result[kh//2:kh//2+H, kw//2:kw//2+W]        image[cy-15:cy+15, cx-15:cx+15] = 0.9



    def frequency_blur(image, kernel):

def create_frame_title(output_dir):

    """Create title frame."""    return image    """Blur using FFT multiplication."""

    fig = plt.figure(figsize=(14, 8))

    ax = fig.add_subplot(111)    H, W = image.shape

    ax.set_xlim(0, 100)

    ax.set_ylim(0, 100)def create_gaussian_kernel(size=15, sigma=2.5):    kh, kw = kernel.shape

    ax.axis('off')

        """Create a normalized Gaussian kernel."""    

    # Title

    ax.text(50, 68, 'THE CONVOLUTION THEOREM', fontsize=32, color=COLORS['accent'],    ax = np.arange(-size // 2 + 1, size // 2 + 1)    # Pad size for linear convolution

            ha='center', va='center', fontweight='bold')

        xx, yy = np.meshgrid(ax, ax)    ph, pw = H + kh - 1, W + kw - 1

    # Main equation

    ax.text(50, 48, r'$f \ast g = \mathcal{F}^{-1}\{\mathcal{F}(f) \cdot \mathcal{F}(g)\}$',    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))    

            fontsize=28, color=COLORS['gold'], ha='center', va='center')

        return kernel / kernel.sum()    # Zero-pad both

    # Subtitle

    ax.text(50, 30, 'Spatial Convolution ≡ Frequency Multiplication',    image_padded = np.zeros((ph, pw), dtype=np.float64)

            fontsize=18, color=COLORS['text'], ha='center', alpha=0.8)

    def spatial_blur(image, kernel):    image_padded[:H, :W] = image

    # Decorative line

    ax.axhline(y=38, xmin=0.2, xmax=0.8, color=COLORS['accent'], linewidth=2, alpha=0.5)    """Blur using scipy spatial convolution."""    

    

    path = os.path.join(output_dir, 'frame_00.png')    from scipy.ndimage import convolve    kernel_padded = np.zeros((ph, pw), dtype=np.float64)

    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.5)

    plt.close()    return convolve(image, kernel, mode='constant')    kernel_padded[:kh, :kw] = kernel

    return path

    



def create_frame_input(output_dir, image, kernel):def frequency_blur(image, kernel):    # FFT, multiply, IFFT

    """Create input frame showing image and kernel."""

    fig = plt.figure(figsize=(14, 8))    """Blur using FFT multiplication."""    F_image = np.fft.fft2(image_padded)

    fig.suptitle('STEP 1: Input Image & Gaussian Kernel', fontsize=22,

                 color=COLORS['accent'], fontweight='bold', y=0.95)    H, W = image.shape    F_kernel = np.fft.fft2(kernel_padded)

    

    gs = fig.add_gridspec(1, 3, width_ratios=[2, 0.3, 1], wspace=0.1)    kh, kw = kernel.shape    result = np.fft.ifft2(F_image * F_kernel).real

    

    ax1 = fig.add_subplot(gs[0])    ph, pw = H + kh - 1, W + kw - 1    

    ax1.imshow(image, cmap='gray', vmin=0, vmax=1)

    ax1.set_title('Input Image I(x,y)', fontsize=14, color=COLORS['text'], pad=10)        # Extract valid region

    ax1.axis('off')

        image_padded = np.zeros((ph, pw), dtype=np.float64)    return result[kh//2:kh//2+H, kw//2:kw//2+W]

    ax_arrow = fig.add_subplot(gs[1])

    ax_arrow.set_xlim(0, 10)    image_padded[:H, :W] = image

    ax_arrow.set_ylim(0, 10)

    ax_arrow.axis('off')    def create_frame(frame_num, image, kernel, spatial_result, freq_result, output_dir):

    ax_arrow.annotate('', xy=(8, 5), xytext=(2, 5),

                      arrowprops=dict(arrowstyle='-|>', color=COLORS['accent'], lw=4))    kernel_padded = np.zeros((ph, pw), dtype=np.float64)    """Create a single frame for the GIF."""

    ax_arrow.text(5, 7, 'BLUR', fontsize=14, color=COLORS['accent'],

                  ha='center', fontweight='bold')    kernel_padded[:kh, :kw] = kernel    

    

    ax2 = fig.add_subplot(gs[2])        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    ax2.imshow(kernel, cmap='hot', interpolation='nearest')

    ax2.set_title(f'Gaussian Kernel ({kernel.shape[0]}×{kernel.shape[1]})',    F_image = np.fft.fft2(image_padded)    fig.suptitle('Convolution Theorem Demonstration', fontsize=16, fontweight='bold')

                  fontsize=14, color=COLORS['text'], pad=10)

    ax2.axis('off')    F_kernel = np.fft.fft2(kernel_padded)    

    

    fig.text(0.5, 0.06, f'Image: {image.shape[0]}×{image.shape[1]} pixels  |  σ = 2.5',    result = np.fft.ifft2(F_image * F_kernel).real    # Frame-specific content

             ha='center', fontsize=13, color=COLORS['text'], alpha=0.7)

            if frame_num == 0:

    plt.tight_layout(rect=[0, 0.1, 1, 0.92])

    path = os.path.join(output_dir, 'frame_01.png')    return result[kh//2:kh//2+H, kw//2:kw//2+W]        # Frame 1: Show original image

    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.3)

    plt.close()        axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)

    return path

def setup_dark_style(fig):        axes[0, 0].set_title('1. Original Image', fontsize=12, fontweight='bold', color='green')



def create_frame_spatial(output_dir, image, kernel, spatial_result):    """Apply dark theme to figure."""        axes[0, 0].axis('off')

    """Create spatial convolution frame."""

    fig = plt.figure(figsize=(14, 8))    fig.patch.set_facecolor(COLORS['bg'])        

    fig.suptitle('STEP 2: Spatial Domain Convolution', fontsize=22,

                 color=COLORS['success'], fontweight='bold', y=0.95)    return fig        for ax in [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]:

    

    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1], hspace=0.2, wspace=0.15)            ax.axis('off')

    

    ax1 = fig.add_subplot(gs[0, 0])def add_glow_text(ax, x, y, text, fontsize=14, color='white', glow_color=None, **kwargs):            ax.set_facecolor('#f0f0f0')

    ax1.imshow(image, cmap='gray', vmin=0, vmax=1)

    ax1.set_title('Input Image', fontsize=13, color=COLORS['text'])    """Add text with subtle glow effect."""        

    ax1.axis('off')

        if glow_color is None:        fig.text(0.5, 0.3, 'Step 1: Load the input image', ha='center', fontsize=14)

    ax2 = fig.add_subplot(gs[0, 1])

    ax2.imshow(kernel, cmap='hot')        glow_color = color        

    ax2.set_title('Kernel', fontsize=13, color=COLORS['text'])

    ax2.axis('off')    # Glow (multiple offset layers)    elif frame_num == 1:

    

    ax3 = fig.add_subplot(gs[0, 2])    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:        # Frame 2: Show kernel

    ax3.imshow(spatial_result, cmap='gray', vmin=0, vmax=1)

    ax3.set_title('Spatial Blur Result', fontsize=13, color=COLORS['success'], fontweight='bold')        ax.text(x + dx*0.5, y + dy*0.5, text, fontsize=fontsize, color=glow_color,         axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)

    ax3.axis('off')

                    alpha=0.15, ha='center', va='center', **kwargs)        axes[0, 0].set_title('Original Image', fontsize=11)

    # Symbols

    fig.text(0.36, 0.62, '⊗', fontsize=40, color=COLORS['success'],    # Main text        axes[0, 0].axis('off')

             ha='center', fontweight='bold')

    fig.text(0.64, 0.62, '=', fontsize=40, color=COLORS['success'],    ax.text(x, y, text, fontsize=fontsize, color=color, ha='center', va='center', **kwargs)        

             ha='center', fontweight='bold')

            axes[0, 1].imshow(kernel, cmap='hot')

    ax_eq = fig.add_subplot(gs[1, :])

    ax_eq.axis('off')def create_title_frame(output_dir, frame_num):        axes[0, 1].set_title('2. Gaussian Kernel (15×15)', fontsize=12, fontweight='bold', color='green')

    ax_eq.text(0.5, 0.6, r'$\mathbf{Output = I \ast K}$ (Direct convolution)',

               fontsize=20, color=COLORS['success'], ha='center', transform=ax_eq.transAxes)    """Create an animated title frame."""        axes[0, 1].axis('off')

    

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])    fig = plt.figure(figsize=(14, 8))        

    path = os.path.join(output_dir, 'frame_02.png')

    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.3)    setup_dark_style(fig)        for ax in [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]:

    plt.close()

    return path    ax = fig.add_subplot(111)            ax.axis('off')



    ax.set_xlim(0, 100)            ax.set_facecolor('#f0f0f0')

def create_frame_fft(output_dir, image, kernel):

    """Create FFT transform frame."""    ax.set_ylim(0, 100)        

    fig = plt.figure(figsize=(14, 8))

    fig.suptitle('STEP 3: Transform to Frequency Domain (FFT)', fontsize=22,    ax.axis('off')        fig.text(0.5, 0.3, 'Step 2: Create Gaussian blur kernel', ha='center', fontsize=14)

                 color=COLORS['purple'], fontweight='bold', y=0.95)

        ax.set_facecolor(COLORS['bg'])        

    F_image = np.fft.fftshift(np.fft.fft2(image))

    kernel_padded = np.zeros_like(image)        elif frame_num == 2:

    kh, kw = kernel.shape

    kernel_padded[:kh, :kw] = kernel    # Main title        # Frame 3: Spatial blur

    F_kernel = np.fft.fftshift(np.fft.fft2(kernel_padded))

        add_glow_text(ax, 50, 70, 'THE CONVOLUTION THEOREM', fontsize=28,         axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)

    gs = fig.add_gridspec(2, 4, height_ratios=[4, 1], hspace=0.25, wspace=0.1)

                      color=COLORS['accent'], glow_color=COLORS['accent'], fontweight='bold')        axes[0, 0].set_title('Original Image', fontsize=11)

    ax1 = fig.add_subplot(gs[0, 0])

    ax1.imshow(image, cmap='gray', vmin=0, vmax=1)            axes[0, 0].axis('off')

    ax1.set_title('Image I', fontsize=12, color=COLORS['text'])

    ax1.axis('off')    # Equation        

    

    ax2 = fig.add_subplot(gs[0, 1])    add_glow_text(ax, 50, 50, r'$f \ast g = \mathcal{F}^{-1}\{\mathcal{F}(f) \cdot \mathcal{F}(g)\}$',         axes[0, 1].imshow(kernel, cmap='hot')

    ax2.imshow(np.log1p(np.abs(F_image)), cmap='magma')

    ax2.set_title('𝓕(I)', fontsize=12, color=COLORS['purple'], fontweight='bold')                  fontsize=24, color=COLORS['gold'])        axes[0, 1].set_title('Gaussian Kernel', fontsize=11)

    ax2.axis('off')

                axes[0, 1].axis('off')

    ax3 = fig.add_subplot(gs[0, 2])

    ax3.imshow(kernel, cmap='hot')    # Subtitle        

    ax3.set_title('Kernel K', fontsize=12, color=COLORS['text'])

    ax3.axis('off')    add_glow_text(ax, 50, 32, 'Spatial Convolution ≡ Frequency Multiplication',         axes[0, 2].imshow(spatial_result, cmap='gray', vmin=0, vmax=1)

    

    ax4 = fig.add_subplot(gs[0, 3])                  fontsize=16, color=COLORS['text'], alpha=0.8)        axes[0, 2].set_title('3. Spatial Convolution', fontsize=12, fontweight='bold', color='green')

    ax4.imshow(np.log1p(np.abs(F_kernel)), cmap='magma')

    ax4.set_title('𝓕(K)', fontsize=12, color=COLORS['purple'], fontweight='bold')            axes[0, 2].axis('off')

    ax4.axis('off')

        # Decorative line        

    # Arrows

    fig.text(0.295, 0.62, '→', fontsize=35, color=COLORS['purple'], fontweight='bold')    ax.plot([20, 80], [42, 42], color=COLORS['accent'], linewidth=2, alpha=0.5)        for ax in [axes[1, 0], axes[1, 1], axes[1, 2]]:

    fig.text(0.275, 0.55, 'FFT', fontsize=11, color=COLORS['purple'])

    fig.text(0.695, 0.62, '→', fontsize=35, color=COLORS['purple'], fontweight='bold')                ax.axis('off')

    fig.text(0.675, 0.55, 'FFT', fontsize=11, color=COLORS['purple'])

        plt.tight_layout()            ax.set_facecolor('#f0f0f0')

    ax_eq = fig.add_subplot(gs[1, :])

    ax_eq.axis('off')    path = os.path.join(output_dir, f'frame_{frame_num:02d}.png')        

    ax_eq.text(0.5, 0.5, 'Fourier Transform: Spatial patterns → Frequency components',

               fontsize=16, color=COLORS['purple'], ha='center', transform=ax_eq.transAxes)    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.5)        fig.text(0.5, 0.3, 'Step 3: Apply spatial convolution (I ⊗ K)', ha='center', fontsize=14)

    

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])    plt.close()        

    path = os.path.join(output_dir, 'frame_03.png')

    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.3)    return path    elif frame_num == 3:

    plt.close()

    return path        # Frame 4: FFT of image



def create_input_frame(output_dir, frame_num, image, kernel):        axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)

def create_frame_multiply(output_dir, image, kernel, freq_result):

    """Create frequency multiplication frame."""    """Show the input image and kernel."""        axes[0, 0].set_title('Original Image', fontsize=11)

    fig = plt.figure(figsize=(14, 8))

    fig.suptitle('STEP 4: Multiply in Frequency Domain → IFFT', fontsize=22,    fig = plt.figure(figsize=(14, 8))        axes[0, 0].axis('off')

                 color=COLORS['gold'], fontweight='bold', y=0.95)

        setup_dark_style(fig)        

    F_image = np.fft.fftshift(np.fft.fft2(image))

    kernel_padded = np.zeros_like(image)            axes[0, 1].imshow(kernel, cmap='hot')

    kh, kw = kernel.shape

    kernel_padded[:kh, :kw] = kernel    # Title        axes[0, 1].set_title('Gaussian Kernel', fontsize=11)

    F_kernel = np.fft.fftshift(np.fft.fft2(kernel_padded))

    F_product = F_image * F_kernel    fig.suptitle('STEP 1: Input Image & Blur Kernel', fontsize=20, color=COLORS['accent'],         axes[0, 1].axis('off')

    

    gs = fig.add_gridspec(2, 4, height_ratios=[4, 1], hspace=0.25, wspace=0.1)                 fontweight='bold', y=0.95)        

    

    ax1 = fig.add_subplot(gs[0, 0])            axes[0, 2].imshow(spatial_result, cmap='gray', vmin=0, vmax=1)

    ax1.imshow(np.log1p(np.abs(F_image)), cmap='magma')

    ax1.set_title('𝓕(I)', fontsize=12, color=COLORS['text'])    gs = fig.add_gridspec(1, 3, width_ratios=[2, 0.5, 1], wspace=0.3)        axes[0, 2].set_title('Spatial Result', fontsize=11)

    ax1.axis('off')

                axes[0, 2].axis('off')

    ax2 = fig.add_subplot(gs[0, 1])

    ax2.imshow(np.log1p(np.abs(F_kernel)), cmap='magma')    # Input image        

    ax2.set_title('𝓕(K)', fontsize=12, color=COLORS['text'])

    ax2.axis('off')    ax1 = fig.add_subplot(gs[0])        F_image = np.fft.fftshift(np.fft.fft2(image))

    

    ax3 = fig.add_subplot(gs[0, 2])    ax1.imshow(image, cmap='gray', vmin=0, vmax=1)        axes[1, 0].imshow(np.log1p(np.abs(F_image)), cmap='magma')

    ax3.imshow(np.log1p(np.abs(F_product)), cmap='magma')

    ax3.set_title('𝓕(I) × 𝓕(K)', fontsize=12, color=COLORS['gold'], fontweight='bold')    ax1.set_title('Input Image I(x,y)', fontsize=14, color=COLORS['text'], pad=10)        axes[1, 0].set_title('4. FFT(Image)', fontsize=12, fontweight='bold', color='green')

    ax3.axis('off')

        ax1.axis('off')        axes[1, 0].axis('off')

    ax4 = fig.add_subplot(gs[0, 3])

    ax4.imshow(freq_result, cmap='gray', vmin=0, vmax=1)    for spine in ax1.spines.values():        

    ax4.set_title('IFFT → Result', fontsize=12, color=COLORS['gold'], fontweight='bold')

    ax4.axis('off')        spine.set_edgecolor(COLORS['accent'])        for ax in [axes[1, 1], axes[1, 2]]:

    

    # Symbols        spine.set_linewidth(2)            ax.axis('off')

    fig.text(0.295, 0.62, '×', fontsize=35, color=COLORS['gold'], fontweight='bold')

    fig.text(0.495, 0.62, '=', fontsize=35, color=COLORS['gold'], fontweight='bold')                ax.set_facecolor('#f0f0f0')

    fig.text(0.695, 0.62, '→', fontsize=35, color=COLORS['gold'], fontweight='bold')

    fig.text(0.68, 0.55, 'IFFT', fontsize=11, color=COLORS['gold'])    # Arrow        

    

    ax_eq = fig.add_subplot(gs[1, :])    ax_arrow = fig.add_subplot(gs[1])        fig.text(0.5, 0.15, 'Step 4: Compute FFT of image → F(I)', ha='center', fontsize=14)

    ax_eq.axis('off')

    ax_eq.text(0.5, 0.5, r'$\mathbf{Output = \mathcal{F}^{-1}\{\mathcal{F}(I) \cdot \mathcal{F}(K)\}}$',    ax_arrow.set_xlim(0, 10)        

               fontsize=20, color=COLORS['gold'], ha='center', transform=ax_eq.transAxes)

        ax_arrow.set_ylim(0, 10)    elif frame_num == 4:

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    path = os.path.join(output_dir, 'frame_04.png')    ax_arrow.axis('off')        # Frame 5: FFT of kernel

    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.3)

    plt.close()    ax_arrow.set_facecolor(COLORS['bg'])        axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)

    return path

    ax_arrow.annotate('', xy=(8, 5), xytext=(2, 5),        axes[0, 0].set_title('Original Image', fontsize=11)



def create_frame_compare(output_dir, spatial_result, freq_result):                     arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=3))        axes[0, 0].axis('off')

    """Create comparison frame."""

    fig = plt.figure(figsize=(14, 8))    ax_arrow.text(5, 7, 'BLUR', fontsize=12, color=COLORS['accent'], ha='center', fontweight='bold')        

    fig.suptitle('STEP 5: Compare Results', fontsize=22,

                 color=COLORS['highlight'], fontweight='bold', y=0.95)            axes[0, 1].imshow(kernel, cmap='hot')

    

    diff = np.abs(spatial_result - freq_result)    # Kernel        axes[0, 1].set_title('Gaussian Kernel', fontsize=11)

    mse = np.mean((spatial_result - freq_result)**2)

        ax2 = fig.add_subplot(gs[2])        axes[0, 1].axis('off')

    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1.5], hspace=0.2, wspace=0.15)

        im = ax2.imshow(kernel, cmap='hot', interpolation='nearest')        

    ax1 = fig.add_subplot(gs[0, 0])

    ax1.imshow(spatial_result, cmap='gray', vmin=0, vmax=1)    ax2.set_title('Gaussian Kernel K', fontsize=14, color=COLORS['text'], pad=10)        axes[0, 2].imshow(spatial_result, cmap='gray', vmin=0, vmax=1)

    ax1.set_title('Spatial Convolution', fontsize=14, color=COLORS['success'], fontweight='bold')

    ax1.axis('off')    ax2.axis('off')        axes[0, 2].set_title('Spatial Result', fontsize=11)

    

    ax2 = fig.add_subplot(gs[0, 1])            axes[0, 2].axis('off')

    ax2.imshow(freq_result, cmap='gray', vmin=0, vmax=1)

    ax2.set_title('Frequency Method', fontsize=14, color=COLORS['gold'], fontweight='bold')    # Info text        

    ax2.axis('off')

        fig.text(0.5, 0.08, f'Image: {image.shape[0]}×{image.shape[1]} pixels  |  Kernel: {kernel.shape[0]}×{kernel.shape[1]}  |  σ = 2.5',        F_image = np.fft.fftshift(np.fft.fft2(image))

    ax3 = fig.add_subplot(gs[0, 2])

    ax3.imshow(diff * 1e14, cmap='hot', vmin=0, vmax=1)             ha='center', fontsize=12, color=COLORS['text'], alpha=0.7)        axes[1, 0].imshow(np.log1p(np.abs(F_image)), cmap='magma')

    ax3.set_title('|Difference| × 10¹⁴', fontsize=14, color=COLORS['text'])

    ax3.axis('off')            axes[1, 0].set_title('FFT(Image)', fontsize=11)

    

    ax_metrics = fig.add_subplot(gs[1, :])    plt.tight_layout(rect=[0, 0.12, 1, 0.92])        axes[1, 0].axis('off')

    ax_metrics.axis('off')

    ax_metrics.text(0.5, 0.7, 'VERIFICATION METRICS', fontsize=16, color=COLORS['text'],    path = os.path.join(output_dir, f'frame_{frame_num:02d}.png')        

                    ha='center', fontweight='bold', transform=ax_metrics.transAxes)

    ax_metrics.text(0.25, 0.3, f'MSE = {mse:.2e}', fontsize=18, color=COLORS['accent'],    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.3)        # Pad kernel for visualization

                    ha='center', fontweight='bold', transform=ax_metrics.transAxes)

    ax_metrics.text(0.5, 0.3, f'Max Diff = {diff.max():.2e}', fontsize=18, color=COLORS['accent'],    plt.close()        kernel_padded = np.zeros_like(image)

                    ha='center', fontweight='bold', transform=ax_metrics.transAxes)

    ax_metrics.text(0.75, 0.3, 'PSNR ≈ ∞', fontsize=18, color=COLORS['accent'],    return path        kh, kw = kernel.shape

                    ha='center', fontweight='bold', transform=ax_metrics.transAxes)

            kernel_padded[:kh, :kw] = kernel

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    path = os.path.join(output_dir, 'frame_05.png')def create_spatial_frame(output_dir, frame_num, image, kernel, spatial_result):        F_kernel = np.fft.fftshift(np.fft.fft2(kernel_padded))

    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.3)

    plt.close()    """Show spatial domain convolution."""        axes[1, 1].imshow(np.log1p(np.abs(F_kernel)), cmap='magma')

    return path

    fig = plt.figure(figsize=(14, 8))        axes[1, 1].set_title('5. FFT(Kernel)', fontsize=12, fontweight='bold', color='green')



def create_frame_conclusion(output_dir, mse):    setup_dark_style(fig)        axes[1, 1].axis('off')

    """Create conclusion frame."""

    fig = plt.figure(figsize=(14, 8))            

    ax = fig.add_subplot(111)

    ax.set_xlim(0, 100)    fig.suptitle('STEP 2: Spatial Domain Convolution', fontsize=20, color=COLORS['success'],         axes[1, 2].axis('off')

    ax.set_ylim(0, 100)

    ax.axis('off')                 fontweight='bold', y=0.95)        axes[1, 2].set_facecolor('#f0f0f0')

    

    # Checkmark circle            

    circle = Circle((50, 62), 12, fill=False, edgecolor=COLORS['success'], linewidth=5)

    ax.add_patch(circle)    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.2)        fig.text(0.5, 0.15, 'Step 5: Compute FFT of kernel → F(K)', ha='center', fontsize=14)

    ax.text(50, 62, '✓', fontsize=55, color=COLORS['success'],

            ha='center', va='center', fontweight='bold')            

    

    # Title    # Input    elif frame_num == 5:

    ax.text(50, 42, 'THEOREM VERIFIED', fontsize=36, color=COLORS['success'],

            ha='center', va='center', fontweight='bold')    ax1 = fig.add_subplot(gs[0, 0])        # Frame 6: Frequency blur result

    

    # Equation box    ax1.imshow(image, cmap='gray', vmin=0, vmax=1)        axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)

    rect = FancyBboxPatch((12, 22), 76, 12, boxstyle="round,pad=0.02",

                          facecolor='none', edgecolor=COLORS['gold'], linewidth=3)    ax1.set_title('Input Image', fontsize=12, color=COLORS['text'])        axes[0, 0].set_title('Original Image', fontsize=11)

    ax.add_patch(rect)

    ax.text(50, 28, r'$I \ast K \equiv \mathcal{F}^{-1}\{\mathcal{F}(I) \cdot \mathcal{F}(K)\}$',    ax1.axis('off')        axes[0, 0].axis('off')

            fontsize=22, color=COLORS['gold'], ha='center', va='center')

                

    # MSE

    ax.text(50, 12, f'MSE = {mse:.2e}  (Machine Precision)', fontsize=18,    # Kernel        axes[0, 1].imshow(kernel, cmap='hot')

            color=COLORS['accent'], ha='center', fontweight='bold')

    ax.text(50, 5, 'Results are IDENTICAL', fontsize=14, color=COLORS['text'],    ax2 = fig.add_subplot(gs[0, 1])        axes[0, 1].set_title('Gaussian Kernel', fontsize=11)

            ha='center', alpha=0.8)

        ax2.imshow(kernel, cmap='hot')        axes[0, 1].axis('off')

    path = os.path.join(output_dir, 'frame_06.png')

    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.5)    ax2.set_title('Kernel', fontsize=12, color=COLORS['text'])        

    plt.close()

    return path    ax2.axis('off')        axes[0, 2].imshow(spatial_result, cmap='gray', vmin=0, vmax=1)



            axes[0, 2].set_title('Spatial Result', fontsize=11)

def main():

    print("🎬 Creating Convolution Theorem Demo GIF...")    # Result        axes[0, 2].axis('off')

    print("=" * 50)

        ax3 = fig.add_subplot(gs[0, 2])        

    output_dir = 'output'

    os.makedirs(output_dir, exist_ok=True)    ax3.imshow(spatial_result, cmap='gray', vmin=0, vmax=1)        F_image = np.fft.fftshift(np.fft.fft2(image))

    

    print("📊 Generating test data...")    ax3.set_title('Spatial Blur Result', fontsize=12, color=COLORS['success'])        axes[1, 0].imshow(np.log1p(np.abs(F_image)), cmap='magma')

    image = create_sample_image(256)

    kernel = create_gaussian_kernel(15, 2.5)    ax3.axis('off')        axes[1, 0].set_title('FFT(Image)', fontsize=11)

    

    print("🔄 Computing blurs...")    for spine in ax3.spines.values():        axes[1, 0].axis('off')

    spatial_result = spatial_blur(image, kernel)

    freq_result = frequency_blur(image, kernel)        spine.set_edgecolor(COLORS['success'])        

    mse = np.mean((spatial_result - freq_result)**2)

    print(f"✅ MSE = {mse:.2e}")        spine.set_linewidth(3)        kernel_padded = np.zeros_like(image)

    

    print("\n🎨 Creating frames...")            kh, kw = kernel.shape

    frames = []

        # Formula        kernel_padded[:kh, :kw] = kernel

    print("  Frame 1/7: Title")

    frames.append(create_frame_title(output_dir))    ax_eq = fig.add_subplot(gs[1, :])        F_kernel = np.fft.fftshift(np.fft.fft2(kernel_padded))

    

    print("  Frame 2/7: Input")    ax_eq.set_xlim(0, 100)        axes[1, 1].imshow(np.log1p(np.abs(F_kernel)), cmap='magma')

    frames.append(create_frame_input(output_dir, image, kernel))

        ax_eq.set_ylim(0, 10)        axes[1, 1].set_title('FFT(Kernel)', fontsize=11)

    print("  Frame 3/7: Spatial")

    frames.append(create_frame_spatial(output_dir, image, kernel, spatial_result))    ax_eq.axis('off')        axes[1, 1].axis('off')

    

    print("  Frame 4/7: FFT")    ax_eq.set_facecolor(COLORS['bg'])        

    frames.append(create_frame_fft(output_dir, image, kernel))

                axes[1, 2].imshow(freq_result, cmap='gray', vmin=0, vmax=1)

    print("  Frame 5/7: Multiply")

    frames.append(create_frame_multiply(output_dir, image, kernel, freq_result))    ax_eq.text(50, 7, r'$\mathbf{Output = I \ast K}$', fontsize=22, color=COLORS['success'],        axes[1, 2].set_title('6. IFFT(F(I)×F(K))', fontsize=12, fontweight='bold', color='green')

    

    print("  Frame 6/7: Compare")               ha='center', va='center', fontweight='bold')        axes[1, 2].axis('off')

    frames.append(create_frame_compare(output_dir, spatial_result, freq_result))

        ax_eq.text(50, 2, 'Direct pixel-by-pixel convolution in spatial domain',         

    print("  Frame 7/7: Conclusion")

    frames.append(create_frame_conclusion(output_dir, mse))               fontsize=12, color=COLORS['text'], ha='center', alpha=0.7)        fig.text(0.5, 0.15, 'Step 6: Multiply in frequency domain → IFFT', ha='center', fontsize=14)

    

    # Load and save GIF            

    pil_frames = [Image.open(f) for f in frames]

    gif_path = os.path.join(output_dir, 'convolution_theorem_demo.gif')    plt.tight_layout(rect=[0, 0.05, 1, 0.92])    elif frame_num == 6:

    

    print(f"\n💾 Saving GIF...")    path = os.path.join(output_dir, f'frame_{frame_num:02d}.png')        # Frame 7: Comparison

    durations = [2500, 2500, 2500, 2500, 2500, 3000, 4000]

        plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.3)        axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)

    pil_frames[0].save(

        gif_path,    plt.close()        axes[0, 0].set_title('Original Image', fontsize=11)

        save_all=True,

        append_images=pil_frames[1:],    return path        axes[0, 0].axis('off')

        duration=durations,

        loop=0,        

        optimize=True

    )def create_fft_frame(output_dir, frame_num, image, kernel):        axes[0, 1].imshow(spatial_result, cmap='gray', vmin=0, vmax=1)

    

    # Cleanup    """Show FFT transformation step."""        axes[0, 1].set_title('Spatial Blur', fontsize=11)

    for f in frames:

        os.remove(f)    fig = plt.figure(figsize=(14, 8))        axes[0, 1].axis('off')

    

    print(f"\n{'=' * 50}")    setup_dark_style(fig)        

    print(f"✨ Done! {gif_path}")

    print(f"   Size: {os.path.getsize(gif_path) / 1024:.1f} KB")            axes[0, 2].imshow(freq_result, cmap='gray', vmin=0, vmax=1)



    fig.suptitle('STEP 3: Transform to Frequency Domain', fontsize=20, color=COLORS['purple'],         axes[0, 2].set_title('Frequency Blur', fontsize=11)

if __name__ == '__main__':

    main()                 fontweight='bold', y=0.95)        axes[0, 2].axis('off')


            

    # Compute FFTs        # Difference (amplified for visibility)

    F_image = np.fft.fftshift(np.fft.fft2(image))        diff = np.abs(spatial_result - freq_result)

    kernel_padded = np.zeros_like(image)        axes[1, 0].imshow(diff * 1e10, cmap='hot', vmin=0, vmax=1)

    kh, kw = kernel.shape        axes[1, 0].set_title('Difference (×10¹⁰)', fontsize=11)

    kernel_padded[:kh, :kw] = kernel        axes[1, 0].axis('off')

    F_kernel = np.fft.fftshift(np.fft.fft2(kernel_padded))        

            # Metrics

    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1], hspace=0.4, wspace=0.15)        axes[1, 1].axis('off')

            mse = np.mean((spatial_result - freq_result)**2)

    # Image and its FFT        axes[1, 1].text(0.5, 0.7, f'MSE = {mse:.2e}', ha='center', va='center', fontsize=16, fontweight='bold')

    ax1 = fig.add_subplot(gs[0, 0])        axes[1, 1].text(0.5, 0.4, '(Machine Precision)', ha='center', va='center', fontsize=12)

    ax1.imshow(image, cmap='gray', vmin=0, vmax=1)        axes[1, 1].set_title('7. Verification', fontsize=12, fontweight='bold', color='green')

    ax1.set_title('Image I', fontsize=11, color=COLORS['text'])        

    ax1.axis('off')        axes[1, 2].axis('off')

            axes[1, 2].text(0.5, 0.5, '✓ THEOREM\nVERIFIED!', ha='center', va='center', 

    ax2 = fig.add_subplot(gs[0, 1])                       fontsize=20, fontweight='bold', color='green')

    ax2.imshow(np.log1p(np.abs(F_image)), cmap='magma')        

    ax2.set_title('FFT(I)', fontsize=11, color=COLORS['purple'])        fig.text(0.5, 0.02, 'Convolution in Space = Multiplication in Frequency', 

    ax2.axis('off')                ha='center', fontsize=14, fontweight='bold', style='italic')

    for spine in ax2.spines.values():    

        spine.set_edgecolor(COLORS['purple'])    plt.tight_layout()

        spine.set_linewidth(2)    frame_path = os.path.join(output_dir, f'frame_{frame_num:02d}.png')

        plt.savefig(frame_path, dpi=100, facecolor='white', bbox_inches='tight')

    # Kernel and its FFT    plt.close()

    ax3 = fig.add_subplot(gs[0, 2])    return frame_path

    ax3.imshow(kernel, cmap='hot')

    ax3.set_title('Kernel K', fontsize=11, color=COLORS['text'])def main():

    ax3.axis('off')    print("Creating demonstration GIF...")

        

    ax4 = fig.add_subplot(gs[0, 3])    output_dir = 'output'

    ax4.imshow(np.log1p(np.abs(F_kernel)), cmap='magma')    os.makedirs(output_dir, exist_ok=True)

    ax4.set_title('FFT(K)', fontsize=11, color=COLORS['purple'])    

    ax4.axis('off')    # Generate data

    for spine in ax4.spines.values():    print("  Generating test image and kernel...")

        spine.set_edgecolor(COLORS['purple'])    image = create_sample_image(256)

        spine.set_linewidth(2)    kernel = create_gaussian_kernel(15, 2.5)

        

    # Arrows    print("  Computing spatial blur...")

    for ax, x_offset in [(ax1, 0.12), (ax3, 0.62)]:    spatial_result = spatial_blur(image, kernel)

        fig.text(x_offset + 0.11, 0.52, '→', fontsize=30, color=COLORS['purple'],     

                 fontweight='bold', transform=fig.transFigure)    print("  Computing frequency blur...")

        fig.text(x_offset + 0.095, 0.47, 'FFT', fontsize=10, color=COLORS['purple'],     freq_result = frequency_blur(image, kernel)

                 transform=fig.transFigure)    

        # Create frames

    # Formula    print("  Creating frames...")

    ax_eq = fig.add_subplot(gs[1, :])    frames = []

    ax_eq.set_xlim(0, 100)    for i in range(7):

    ax_eq.set_ylim(0, 10)        frame_path = create_frame(i, image, kernel, spatial_result, freq_result, output_dir)

    ax_eq.axis('off')        frames.append(Image.open(frame_path))

    ax_eq.set_facecolor(COLORS['bg'])        print(f"    Frame {i+1}/7 created")

        

    ax_eq.text(50, 6, r'$\mathcal{F}(I) \rightarrow$ Frequency Spectrum  |  $\mathcal{F}(K) \rightarrow$ Frequency Response',     # Create GIF

               fontsize=14, color=COLORS['purple'], ha='center')    gif_path = os.path.join(output_dir, 'convolution_theorem_demo.gif')

    ax_eq.text(50, 1, 'Fourier Transform converts spatial patterns to frequency components',     print(f"  Saving GIF to {gif_path}...")

               fontsize=11, color=COLORS['text'], ha='center', alpha=0.7)    

        # Durations: longer pause on first and last frames

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])    durations = [1500, 1200, 1200, 1200, 1200, 1200, 3000]  # milliseconds

    path = os.path.join(output_dir, f'frame_{frame_num:02d}.png')    

    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.3)    frames[0].save(

    plt.close()        gif_path,

    return path        save_all=True,

        append_images=frames[1:],

def create_multiply_frame(output_dir, frame_num, image, kernel, freq_result):        duration=durations,

    """Show frequency multiplication and IFFT."""        loop=0

    fig = plt.figure(figsize=(14, 8))    )

    setup_dark_style(fig)    

        # Clean up frame files

    fig.suptitle('STEP 4: Multiply in Frequency Domain → IFFT', fontsize=20, color=COLORS['gold'],     for i in range(7):

                 fontweight='bold', y=0.95)        os.remove(os.path.join(output_dir, f'frame_{i:02d}.png'))

        

    # Compute FFTs    print(f"\n✓ GIF created: {gif_path}")

    F_image = np.fft.fftshift(np.fft.fft2(image))    print(f"  Size: {os.path.getsize(gif_path) / 1024:.1f} KB")

    kernel_padded = np.zeros_like(image)    

    kh, kw = kernel.shape    # Verify

    kernel_padded[:kh, :kw] = kernel    mse = np.mean((spatial_result - freq_result)**2)

    F_kernel = np.fft.fftshift(np.fft.fft2(kernel_padded))    print(f"  MSE: {mse:.2e}")

    F_product = F_image * F_kernel

    if __name__ == '__main__':

    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1], hspace=0.4, wspace=0.15)    main()

    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.log1p(np.abs(F_image)), cmap='magma')
    ax1.set_title('FFT(I)', fontsize=11, color=COLORS['text'])
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.log1p(np.abs(F_kernel)), cmap='magma')
    ax2.set_title('FFT(K)', fontsize=11, color=COLORS['text'])
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(np.log1p(np.abs(F_product)), cmap='magma')
    ax3.set_title('FFT(I) × FFT(K)', fontsize=11, color=COLORS['gold'])
    ax3.axis('off')
    for spine in ax3.spines.values():
        spine.set_edgecolor(COLORS['gold'])
        spine.set_linewidth(2)
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(freq_result, cmap='gray', vmin=0, vmax=1)
    ax4.set_title('IFFT → Result', fontsize=11, color=COLORS['gold'])
    ax4.axis('off')
    for spine in ax4.spines.values():
        spine.set_edgecolor(COLORS['gold'])
        spine.set_linewidth(3)
    
    # Symbols between plots
    fig.text(0.28, 0.58, '×', fontsize=30, color=COLORS['gold'], fontweight='bold', transform=fig.transFigure)
    fig.text(0.51, 0.58, '=', fontsize=30, color=COLORS['gold'], fontweight='bold', transform=fig.transFigure)
    fig.text(0.72, 0.55, '→', fontsize=30, color=COLORS['gold'], fontweight='bold', transform=fig.transFigure)
    fig.text(0.71, 0.50, 'IFFT', fontsize=10, color=COLORS['gold'], transform=fig.transFigure)
    
    # Formula
    ax_eq = fig.add_subplot(gs[1, :])
    ax_eq.set_xlim(0, 100)
    ax_eq.set_ylim(0, 10)
    ax_eq.axis('off')
    ax_eq.set_facecolor(COLORS['bg'])
    
    ax_eq.text(50, 6, r'$\mathbf{Output = \mathcal{F}^{-1}\{\mathcal{F}(I) \cdot \mathcal{F}(K)\}}$', 
               fontsize=20, color=COLORS['gold'], ha='center', fontweight='bold')
    ax_eq.text(50, 1, 'Inverse FFT transforms back to spatial domain', 
               fontsize=11, color=COLORS['text'], ha='center', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    path = os.path.join(output_dir, f'frame_{frame_num:02d}.png')
    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.3)
    plt.close()
    return path

def create_comparison_frame(output_dir, frame_num, spatial_result, freq_result):
    """Show final comparison proving equivalence."""
    fig = plt.figure(figsize=(14, 8))
    setup_dark_style(fig)
    
    fig.suptitle('STEP 5: Verify Equivalence', fontsize=20, color=COLORS['highlight'], 
                 fontweight='bold', y=0.95)
    
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.5], hspace=0.3, wspace=0.2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(spatial_result, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Spatial Convolution', fontsize=13, color=COLORS['success'], fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(freq_result, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Frequency Multiplication', fontsize=13, color=COLORS['gold'], fontweight='bold')
    ax2.axis('off')
    
    # Difference (amplified)
    diff = np.abs(spatial_result - freq_result)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(diff * 1e14, cmap='hot', vmin=0, vmax=1)
    ax3.set_title('Difference (×10¹⁴)', fontsize=13, color=COLORS['text'])
    ax3.axis('off')
    
    # Metrics panel
    ax_metrics = fig.add_subplot(gs[1, :])
    ax_metrics.set_xlim(0, 100)
    ax_metrics.set_ylim(0, 20)
    ax_metrics.axis('off')
    ax_metrics.set_facecolor(COLORS['bg'])
    
    mse = np.mean((spatial_result - freq_result)**2)
    
    ax_metrics.text(50, 16, 'VERIFICATION METRICS', fontsize=14, color=COLORS['text'], 
                   ha='center', fontweight='bold')
    
    ax_metrics.text(25, 10, f'MSE = {mse:.2e}', fontsize=16, color=COLORS['accent'], 
                   ha='center', fontweight='bold')
    ax_metrics.text(50, 10, f'Max Diff = {diff.max():.2e}', fontsize=16, color=COLORS['accent'], 
                   ha='center', fontweight='bold')
    ax_metrics.text(75, 10, 'PSNR = ∞ dB', fontsize=16, color=COLORS['accent'], 
                   ha='center', fontweight='bold')
    
    ax_metrics.text(50, 3, '(Machine Precision - Results are IDENTICAL)', 
                   fontsize=12, color=COLORS['text'], ha='center', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    path = os.path.join(output_dir, f'frame_{frame_num:02d}.png')
    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.3)
    plt.close()
    return path

def create_conclusion_frame(output_dir, frame_num, mse):
    """Create final conclusion frame."""
    fig = plt.figure(figsize=(14, 8))
    setup_dark_style(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    
    # Success checkmark circle
    circle = Circle((50, 65), 15, fill=False, edgecolor=COLORS['success'], linewidth=4)
    ax.add_patch(circle)
    ax.text(50, 65, '✓', fontsize=50, color=COLORS['success'], ha='center', va='center', fontweight='bold')
    
    # Main message
    add_glow_text(ax, 50, 42, 'THEOREM VERIFIED', fontsize=32, 
                  color=COLORS['success'], glow_color=COLORS['success'], fontweight='bold')
    
    # Equation box
    ax.add_patch(FancyBboxPatch((15, 22), 70, 12, boxstyle="round,pad=0.02",
                                 facecolor='none', edgecolor=COLORS['gold'], linewidth=2))
    ax.text(50, 28, r'$I \ast K \equiv \mathcal{F}^{-1}\{\mathcal{F}(I) \cdot \mathcal{F}(K)\}$', 
            fontsize=20, color=COLORS['gold'], ha='center', va='center')
    
    # MSE result
    ax.text(50, 12, f'MSE = {mse:.2e}', fontsize=18, color=COLORS['accent'], ha='center', fontweight='bold')
    ax.text(50, 5, 'Spatial Domain = Frequency Domain', fontsize=14, color=COLORS['text'], 
            ha='center', alpha=0.8)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f'frame_{frame_num:02d}.png')
    plt.savefig(path, dpi=120, facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.5)
    plt.close()
    return path

def main():
    print("🎬 Creating Convolution Theorem Demo GIF...")
    print("=" * 50)
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    print("📊 Generating test data...")
    image = create_sample_image(256)
    kernel = create_gaussian_kernel(15, 2.5)
    
    print("🔄 Computing spatial blur...")
    spatial_result = spatial_blur(image, kernel)
    
    print("🔄 Computing frequency blur...")
    freq_result = frequency_blur(image, kernel)
    
    mse = np.mean((spatial_result - freq_result)**2)
    print(f"✅ MSE = {mse:.2e}")
    
    # Create frames
    print("\n🎨 Creating frames...")
    frame_paths = []
    
    print("  Frame 1/6: Title")
    frame_paths.append(create_title_frame(output_dir, 0))
    
    print("  Frame 2/6: Input")
    frame_paths.append(create_input_frame(output_dir, 1, image, kernel))
    
    print("  Frame 3/6: Spatial Convolution")
    frame_paths.append(create_spatial_frame(output_dir, 2, image, kernel, spatial_result))
    
    print("  Frame 4/6: FFT Transform")
    frame_paths.append(create_fft_frame(output_dir, 3, image, kernel))
    
    print("  Frame 5/6: Frequency Multiplication")
    frame_paths.append(create_multiply_frame(output_dir, 4, image, kernel, freq_result))
    
    print("  Frame 6/6: Comparison")
    frame_paths.append(create_comparison_frame(output_dir, 5, spatial_result, freq_result))
    
    print("  Frame 7/7: Conclusion")
    frame_paths.append(create_conclusion_frame(output_dir, 6, mse))
    
    # Load frames
    frames = [Image.open(p) for p in frame_paths]
    
    # Create GIF with smooth timing
    gif_path = os.path.join(output_dir, 'convolution_theorem_demo.gif')
    print(f"\n💾 Saving GIF: {gif_path}")
    
    # Frame durations (ms): Title, Input, Spatial, FFT, Multiply, Compare, Conclusion
    durations = [2000, 2500, 2500, 2500, 2500, 3000, 4000]
    
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True
    )
    
    # Cleanup temp frames
    for p in frame_paths:
        os.remove(p)
    
    print(f"\n{'=' * 50}")
    print(f"✨ GIF created successfully!")
    print(f"   📁 Path: {gif_path}")
    print(f"   📐 Size: {os.path.getsize(gif_path) / 1024:.1f} KB")
    print(f"   🎞️  Frames: {len(frames)}")
    print(f"   ⏱️  Duration: ~{sum(durations)/1000:.1f}s per loop")

if __name__ == '__main__':
    main()
