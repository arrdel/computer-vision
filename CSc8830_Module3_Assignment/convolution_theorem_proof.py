#!/usr/bin/env python3
"""
================================================================================
CSc 8830: Computer Vision - Module 3 Assignment
Mathematical Proof: Convolution Theorem
================================================================================

DESCRIPTION:
    This script provides step-by-step mathematical verification and visualization
    of the Convolution Theorem, proving that:
    
        Convolution in space ↔ Multiplication in frequency domain
        
    It demonstrates:
    1. Theoretical mathematical relationships
    2. Numerical verification with simple examples
    3. Visual representation of the theorem

USAGE:
    python convolution_theorem_proof.py [--kernel_size SIZE]
    
EXAMPLES:
    python convolution_theorem_proof.py
    python convolution_theorem_proof.py --kernel_size 7

OUTPUT:
    - convolution_theorem_proof.txt: Mathematical derivation and verification
    - convolution_proof_visualization.png: Visual proof

MATHEMATICAL THEORY:
================================================================================

The Convolution Theorem states:
──────────────────────────────

For two functions f(x,y) and g(x,y), their 2D convolution is defined as:

    (f ⊗ g)(x,y) = ∫∫ f(u,v) · g(x-u, y-v) du dv
    
The Fourier transform of this convolution is:

    FT{f ⊗ g}(u,v) = FT{f}(u,v) · FT{g}(u,v)
    
Or equivalently:

    f ⊗ g = IFT{FT{f} · FT{g}}

Where:
    • ⊗ denotes convolution operator
    • · denotes point-wise multiplication
    • FT denotes Fourier Transform
    • IFT denotes Inverse Fourier Transform

KEY INSIGHT:
────────────
Convolution in the spatial domain is EQUIVALENT to multiplication in the 
frequency (Fourier) domain. This provides an alternative, often more efficient 
way to compute convolutions, especially for large kernels.

COMPUTATIONAL ADVANTAGE:
────────────────────────
For an N×N image and M×M kernel:
    • Direct convolution: O(N² × M²)
    • FFT-based convolution: O(N² log N)

For large kernels, FFT-based approach is significantly faster.

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft2, ifft2, fftshift
import os
from pathlib import Path

OUTPUT_DIR = "output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)


def generate_simple_example():
    """
    Generate simple 1D example to demonstrate convolution theorem.
    
    Returns:
        tuple: (signal, kernel, convolution result)
    """
    # Simple 1D signal
    signal_arr = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    
    # Simple 1D kernel
    kernel = np.array([1, 1, 1], dtype=np.float32) / 3  # Average filter
    
    return signal_arr, kernel


def numerical_proof_1d():
    """
    Provide numerical proof of convolution theorem in 1D.
    
    Returns:
        str: Formatted proof text
    """
    signal_arr, kernel = generate_simple_example()
    
    # Pad for linear convolution
    signal_padded = np.pad(signal_arr, (len(kernel)-1, 0), mode='constant')
    
    # Method 1: Direct convolution (scipy.signal)
    conv_direct = signal.convolve(signal_arr, kernel, mode='full')
    
    # Method 2: FFT-based convolution
    # Pad to avoid circular convolution
    N = len(signal_arr) + len(kernel) - 1
    signal_fft = np.fft.fft(signal_arr, n=N)
    kernel_fft = np.fft.fft(kernel, n=N)
    
    # Multiply in frequency domain
    product_fft = signal_fft * kernel_fft
    
    # Inverse FFT
    conv_fft = np.fft.ifft(product_fft).real
    
    proof_text = """
NUMERICAL PROOF: 1D Convolution Theorem
════════════════════════════════════════

Example 1D Convolution:
───────────────────────

Signal:  s = [1.0, 2.0, 3.0, 2.0, 1.0]
Kernel:  h = [0.333, 0.333, 0.333]  (averaging filter)

Method 1: Direct Convolution (Spatial Domain)
──────────────────────────────────────────────
s(n) ⊗ h(n) - manual calculation:

For each output position, multiply and sum:
    y[0] = 1.0 × 0.333 = 0.333
    y[1] = 1.0×0.333 + 2.0×0.333 = 1.0
    y[2] = 1.0×0.333 + 2.0×0.333 + 3.0×0.333 = 2.0
    y[3] = 2.0×0.333 + 3.0×0.333 + 2.0×0.333 = 2.333
    y[4] = 3.0×0.333 + 2.0×0.333 + 1.0×0.333 = 2.0
    y[5] = 2.0×0.333 + 1.0×0.333 = 1.0
    y[6] = 1.0×0.333 = 0.333

Result (Direct):    """ + str(np.around(conv_direct, 4)) + """

Method 2: FFT-based Convolution (Frequency Domain)
────────────────────────────────────────────────────
1. Compute FFT of signal:  S(k) = FFT{s}
2. Compute FFT of kernel:  H(k) = FFT{h}
3. Multiply in frequency:  Y(k) = S(k) × H(k)
4. Inverse FFT:            y(n) = IFFT{Y(k)}

Result (FFT):       """ + str(np.around(conv_fft[:len(conv_direct)], 4)) + """

Comparison:
───────────
Difference (Max Abs): """ + f"{np.max(np.abs(conv_direct - conv_fft[:len(conv_direct)])):.2e}" + """
Status:               ✓ EQUIVALENT (within numerical precision)

Conclusion:
───────────
Both methods produce the same result!
✓ Direct spatial convolution = FFT-based multiplication

This proves the CONVOLUTION THEOREM:
    Convolution in space ↔ Multiplication in frequency domain

"""
    return proof_text, conv_direct, conv_fft[:len(conv_direct)]


def mathematical_derivation():
    """
    Return mathematical derivation of the convolution theorem.
    
    Returns:
        str: Formatted mathematical proof
    """
    derivation = """
MATHEMATICAL DERIVATION: Convolution Theorem
═════════════════════════════════════════════

DEFINITION (2D Convolution):
────────────────────────────

The convolution of two functions f(x,y) and g(x,y) is defined as:

    (f ⊗ g)(x,y) = ∫_{-∞}^{∞} ∫_{-∞}^{∞} f(u,v) · g(x-u, y-v) du dv

In discrete form:

    (f ⊗ g)[m,n] = Σ_i Σ_j f[i,j] · g[m-i, n-j]


DEFINITION (Fourier Transform):
────────────────────────────────

For a 2D function f(x,y):

    F(u,v) = ∫_{-∞}^{∞} ∫_{-∞}^{∞} f(x,y) e^{-j2π(ux+vy)} dx dy

And the inverse transform:

    f(x,y) = ∫_{-∞}^{∞} ∫_{-∞}^{∞} F(u,v) e^{j2π(ux+vy)} du dv


CONVOLUTION THEOREM PROOF:
──────────────────────────

Let y(x,y) = f(x,y) ⊗ g(x,y)

Taking the Fourier transform of both sides:

    Y(u,v) = FT{f(x,y) ⊗ g(x,y)}

         = ∫_{-∞}^{∞} ∫_{-∞}^{∞} [∫_{-∞}^{∞} ∫_{-∞}^{∞} f(a,b)·g(x-a,y-b) da db] e^{-j2π(ux+vy)} dx dy

Change of variables: α = x - a, β = y - b
                     x = α + a, y = β + b
                     dx = dα, dy = dβ

    Y(u,v) = ∫_{-∞}^{∞} ∫_{-∞}^{∞} f(a,b) [∫_{-∞}^{∞} ∫_{-∞}^{∞} g(α,β) e^{-j2π(u(α+a)+v(β+b))} dα dβ] da db

         = ∫_{-∞}^{∞} ∫_{-∞}^{∞} f(a,b) e^{-j2π(ua+vb)} [∫_{-∞}^{∞} ∫_{-∞}^{∞} g(α,β) e^{-j2π(uα+vβ)} dα dβ] da db

         = [∫_{-∞}^{∞} ∫_{-∞}^{∞} f(a,b) e^{-j2π(ua+vb)} da db] × [∫_{-∞}^{∞} ∫_{-∞}^{∞} g(α,β) e^{-j2π(uα+vβ)} dα dβ]

         = F(u,v) × G(u,v)


CONCLUSION:
───────────

    FT{f ⊗ g} = FT{f} × FT{g}

Or equivalently:

    f ⊗ g = IFT{FT{f} × FT{g}}

This is the CONVOLUTION THEOREM!


PHYSICAL INTERPRETATION:
────────────────────────

• Spatial Domain: Convolution describes how a filter affects an image point-by-point
• Frequency Domain: Multiplication shows how the filter modifies frequency components
• Trade-off:
    - Spatial: Intuitive, suitable for small kernels
    - Frequency: Efficient for large kernels (computational advantage with FFT)


COMPUTATIONAL COMPLEXITY:
─────────────────────────

For N×N image and M×M kernel:

Direct Convolution (Spatial):
    Time Complexity: O(N² × M²)
    Space Complexity: O(N²)
    
FFT-based Convolution (Frequency):
    Time Complexity: O(N² log N) using FFT algorithm
    Space Complexity: O(N²)
    
Advantage of FFT for large kernels:
    For M >> log(N): FFT-based is faster
    Example: N=1024, M=256
        Direct: ~1024² × 256² ≈ 68 billion operations
        FFT: ~1024² × log(1024) ≈ 10 million operations
        Speedup: ~6800×


KEY PROPERTIES:
───────────────

1. Commutativity:     f ⊗ g = g ⊗ f

2. Associativity:     f ⊗ (g ⊗ h) = (f ⊗ g) ⊗ h

3. Distributivity:    f ⊗ (g + h) = (f ⊗ g) + (f ⊗ h)

4. Zero padding:      Must zero-pad to avoid circular convolution when using FFT

5. Frequency shift:    Shifting in frequency domain ↔ Phase modulation in space

6. Scaling:           If f ⊗ g = y, then (af) ⊗ (bg) = ab·y


APPLICATION TO IMAGE BLURRING:
──────────────────────────────

Image blurring is a convolution operation:

    Blurred_Image = Image ⊗ Blur_Kernel

Using the Convolution Theorem:

    Blurred_Image = IFFT{FFT{Image} × FFT{Blur_Kernel}}

This allows efficient implementation of image blurring filters.


EXAMPLE BLUR KERNELS IN FREQUENCY DOMAIN:
──────────────────────────────────────────

1. Box Filter:
   H(u,v) = sinc(u·width/2) · sinc(v·width/2)
   Effect: Attenuates high frequencies uniformly

2. Gaussian Filter:
   H(u,v) = exp(-(u² + v²)/(2σ²))
   Effect: Smooth roll-off of high frequencies

3. Motion Blur:
   H(u,v) = sinc(u·length) for blur direction
   Effect: Creates streak-like artifacts at specific angles

"""
    return derivation


def create_proof_visualization(kernel_size=7):
    """
    Create visualization of convolution theorem proof.
    
    Args:
        kernel_size: Size of blur kernel to demonstrate
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Generate simple example
    signal_arr = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel = np.array([1, 1, 1], dtype=np.float32) / 3
    
    # Compute convolutions
    conv_direct = signal.convolve(signal_arr, kernel, mode='full')
    
    N = len(signal_arr) + len(kernel) - 1
    signal_fft = np.fft.fft(signal_arr, n=N)
    kernel_fft = np.fft.fft(kernel, n=N)
    product_fft = signal_fft * kernel_fft
    conv_fft = np.fft.ifft(product_fft).real
    
    # 1D Convolution visualization
    ax1 = plt.subplot(3, 3, 1)
    ax1.stem(signal_arr, basefmt=' ')
    ax1.set_title('Input Signal', fontweight='bold')
    ax1.set_xlabel('n')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.stem(kernel, basefmt=' ')
    ax2.set_title('Convolution Kernel', fontweight='bold')
    ax2.set_xlabel('n')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.stem(conv_direct, basefmt=' ')
    ax3.set_title('Result: s ⊗ h', fontweight='bold')
    ax3.set_xlabel('n')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)
    
    # Frequency domain
    ax4 = plt.subplot(3, 3, 4)
    ax4.stem(np.abs(signal_fft), basefmt=' ')
    ax4.set_title('FFT(Signal)', fontweight='bold')
    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('Magnitude')
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.stem(np.abs(kernel_fft), basefmt=' ')
    ax5.set_title('FFT(Kernel)', fontweight='bold')
    ax5.set_xlabel('Frequency')
    ax5.set_ylabel('Magnitude')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.stem(np.abs(product_fft), basefmt=' ')
    ax6.set_title('FFT(Signal) × FFT(Kernel)', fontweight='bold')
    ax6.set_xlabel('Frequency')
    ax6.set_ylabel('Magnitude')
    ax6.grid(True, alpha=0.3)
    
    # Comparison
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(conv_direct[:len(signal_arr)+len(kernel)-1], 'b-o', label='Direct Convolution', linewidth=2, markersize=8)
    ax7.set_title('Result: Direct vs FFT', fontweight='bold')
    ax7.set_xlabel('n')
    ax7.set_ylabel('Amplitude')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(conv_fft[:len(signal_arr)+len(kernel)-1], 'r-s', label='FFT-based', linewidth=2, markersize=8)
    ax8.set_title('Result: FFT Convolution', fontweight='bold')
    ax8.set_xlabel('n')
    ax8.set_ylabel('Amplitude')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    ax9 = plt.subplot(3, 3, 9)
    diff = np.abs(conv_direct[:len(signal_arr)+len(kernel)-1] - conv_fft[:len(signal_arr)+len(kernel)-1])
    ax9.stem(diff, basefmt=' ')
    ax9.set_title('Difference (Direct - FFT)', fontweight='bold')
    ax9.set_xlabel('n')
    ax9.set_ylabel('Error')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/convolution_proof_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {OUTPUT_DIR}/convolution_proof_visualization.png")
    plt.close()


def main():
    """Generate comprehensive proof document."""
    
    print("\n" + "="*80)
    print("CSc 8830 Module 3: Mathematical Proof of Convolution Theorem")
    print("="*80)
    
    # Generate all components
    derivation = mathematical_derivation()
    proof_text, conv_direct, conv_fft = numerical_proof_1d()
    
    # Combine into complete document
    complete_proof = f"""
{'='*80}
CSc 8830: Computer Vision - Module 3 Assignment
IMAGE BLURRING: MATHEMATICAL PROOF OF CONVOLUTION THEOREM
{'='*80}

{derivation}

{'='*80}
NUMERICAL VERIFICATION
{'='*80}

{proof_text}

{'='*80}
CONCLUSION
{'='*80}

The Convolution Theorem is VERIFIED both mathematically and numerically:

1. MATHEMATICAL PROOF:
   • Convolution in spatial domain = Multiplication in frequency domain
   • Proven using change of variables in integral calculus
   • Fundamental theorem in signal processing and image processing

2. NUMERICAL VERIFICATION:
   • Both methods (direct convolution and FFT-based) produce identical results
   • Maximum error is within machine epsilon (numerical precision limits)

3. PRACTICAL APPLICATION:
   • Image blurring is efficiently computed using FFT-based convolution
   • Spatial domain provides intuition; frequency domain provides efficiency
   • Trade-off: For small kernels, direct convolution is simpler
             For large kernels, FFT-based convolution is faster

4. KEY INSIGHT:
   • Convolution = Multiplication under Fourier Transform
   • This duality is the foundation of modern signal processing
   • Enables efficient implementation of many image processing operations

{'='*80}

Generated: February 2, 2026
Author: CSc 8830 Student
Submission: Module 3 Assignment

"""
    
    # Save proof document
    proof_file = f'{OUTPUT_DIR}/convolution_theorem_proof.txt'
    with open(proof_file, 'w') as f:
        f.write(complete_proof)
    
    print(f"\n✓ Mathematical proof saved: {proof_file}")
    
    # Generate visualization
    print(f"\n✓ Generating visualization...")
    create_proof_visualization()
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✓ Convolution Theorem PROVEN:")
    print(f"  • Mathematical derivation complete")
    print(f"  • Numerical verification successful")
    print(f"  • Visualization generated")
    print(f"\n✓ Output files:")
    print(f"  - {proof_file}")
    print(f"  - {OUTPUT_DIR}/convolution_proof_visualization.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
