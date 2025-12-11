# 🧬 Interactive Protein Folding Visualization

An interactive 3D visualization tool for exploring protein folding dynamics and secondary structures. Built with Three.js, this single-file web application provides an intuitive interface to understand how proteins transition from unfolded linear chains to complex 3D structures.

![Protein Folding Visualization](https://img.shields.io/badge/Status-Active-success)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)
![Three.js](https://img.shields.io/badge/Three.js-000000?logo=three.js&logoColor=white)

## ✨ Features

- **Interactive 3D Visualization**: Rotate, zoom, and pan to explore protein structures from any angle
- **Folding Animation**: Watch proteins fold from unfolded linear chains to complex 3D structures
- **Secondary Structure Display**: 
  - 🔴 **α-Helix** regions (red)
  - 🔵 **β-Sheet** regions (cyan)
  - 🟡 **Random Coil** regions (yellow)
- **Real-time Controls**:
  - Adjust folding progress manually
  - Control animation speed
  - Toggle structure types on/off
  - Wireframe mode
  - Adjustable atom size
- **Protein Statistics**: View counts of residues, helices, sheets, and coils
- **Random Protein Generation**: Generate new protein structures with different conformations

## 🚀 Quick Start

1. **Clone or download** this repository
2. **Open** `protein_folding.html` in any modern web browser
3. **No installation required** - it's a single self-contained HTML file!

```bash
# Clone the repository
git clone https://github.com/eytanmerkin/my_practice_project_1.git
cd my_practice_project_1

# Open in browser
open protein_folding.html  # macOS
# or
xdg-open protein_folding.html  # Linux
# or simply double-click the file
```

## 🎮 How to Use

### Mouse Controls
- **Left Click + Drag**: Rotate the protein structure
- **Scroll Wheel**: Zoom in/out
- **Right Click + Drag**: Pan the view (if needed)

### Controls Panel

#### Folding Animation
- **Progress Slider**: Manually adjust folding progress (0% = unfolded, 100% = folded)
- **Speed Slider**: Control animation playback speed
- **▶ Play Button**: Start/pause the folding animation
- **↻ Reset**: Return to unfolded state
- **🎲 New**: Generate a random protein structure

#### Secondary Structure
Toggle visibility of:
- α-Helix regions
- β-Sheet regions
- Random Coil regions

#### View Options
- **Wireframe**: Switch between solid and wireframe rendering
- **Center View**: Reset camera to default position
- **Atom Size**: Adjust the size of atom spheres

## 🔬 How the Folding Model Works

### Current Implementation

This visualization uses a **simplified geometric model** to demonstrate protein folding concepts:

1. **Structure Generation**:
   - Randomly assigns secondary structure regions (helix, sheet, or coil)
   - Each region has realistic lengths (8-20 residues for helices, 5-13 for sheets)
   - Structures are generated using geometric rules based on known protein structure parameters

2. **3D Positioning**:
   - **α-Helix**: Spiral structure with 3.6 residues per turn, ~1.5Å rise per residue
   - **β-Sheet**: Extended pleated structure with ~3.4Å spacing between residues
   - **Random Coil**: Flexible regions using random walk algorithm

3. **Folding Animation**:
   - Linear interpolation between unfolded (straight chain) and folded (3D structure) states
   - Smooth transition showing the folding process

### Important Note

⚠️ **This is a visualization tool, not a prediction engine.**

Real protein folding prediction (like AlphaFold) uses:
- Deep learning models trained on thousands of known structures
- Amino acid sequence analysis
- Evolutionary information from related proteins
- Physical forces (hydrophobic interactions, hydrogen bonds, electrostatic forces)
- Energy minimization algorithms

This tool demonstrates the **concept** of protein folding and secondary structures, but does not predict how a specific amino acid sequence would actually fold.

## 🛠️ Technical Details

### Technologies Used
- **Three.js r128**: 3D graphics rendering
- **Vanilla JavaScript**: No frameworks required
- **HTML5/CSS3**: Modern web standards
- **WebGL**: Hardware-accelerated 3D rendering

### Browser Compatibility
- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Opera

Requires a modern browser with WebGL support.

### Performance
- Optimized mesh reuse (no recreation on each frame)
- Efficient memory management
- Smooth 60fps animation
- Responsive controls

## 📁 Project Structure

```
my_practice_project_1/
├── protein_folding.html  # Main visualization application
├── targil.py             # Python file (if applicable)
└── README.md             # This file
```

## 🎨 Color Scheme

- **α-Helix**: `#ff6b6b` (Red)
- **β-Sheet**: `#4ecdc4` (Cyan)
- **Random Coil**: `#ffd93d` (Yellow)
- **Bonds**: White with transparency

## 🔮 Future Enhancements

Potential improvements:
- [ ] Load real PDB (Protein Data Bank) files
- [ ] Amino acid sequence input
- [ ] More realistic folding pathways
- [ ] Export structures as images
- [ ] Multiple protein comparison view
- [ ] Energy landscape visualization

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for exploring the fascinating world of protein structures**

