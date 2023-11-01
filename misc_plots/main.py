from rdkit import Chem
import os
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Create a molecule object from the SMILES strings
smiles = list(
    set(
        [
            "NC1CNC2=C1CCC2",
            "CCC1C2=C3NC321",
            "CN1CN1C",
            "CCC1CC1",
            "CCC(C)C1C(C)C1N",
            "CC1CC(O)N1C",
            "CCC1C(N)C1(C)N",
            "CC1C2C34CC35C1C245",
            "C=CCCCC",
            "CN(O)CO",
            "OC12CCC13C=C23",
            "NC12CN=C3CC1(C3)C2",
            "CNC1OC1C1COC1",
            "COC1(C)C23CC24CC413",
            "C=CC1CC1C",
            "CN(C)CO",
            "CNC1OC1C1COC1",
            "CC1C2CC12C",
            "CNC1COC(C)(C)C1",
            "CC1(O)NOO1",
            "CCC(C)O",
            "COC1(C)C23CC24CC413",
            "C=CC1CC1C",
            "CN(C)CO",
        ]
    )
)
smiles = list(
    set(
        [
            "CC1NC1c1ncon1",
            "CC1C(C)N(C)C1C=O",
            "CC1(C)CC1NC(N)=O",
            "C#CC1(C(C)OC)CO1",
            "CCC(C#N)(C=O)CC",
            "CC1C(O)CC1C1CC1",
            "CC1OC1(CO)CC#N",
            "CC1NC12CC(C)(O)C2",
            "OC12CCC1CC1CC12",
            "CC12C(=O)C3C(O)C1N32",
            "c1n[nH]c(CC2CO2)n1",
            "N#CC(N)CC(F)(F)F",
            "CC12CC(O1)C1(CN1)C2",
            "O=C1C=CNC1=O",
            "Cn1ncnc1CO",
            "OCC1OC2CCC2O1",
            "Cc1nc2c[nH]cc2o1",
            "NC=NCCCC(=O)O",
            "C#CC1CC(C=O)C1C",
            "COC(=O)C1CN1",
        ]
    )
)

molecules = [Chem.MolFromSmiles(s) for s in smiles]

# Save individual molecule images
image_paths = []
for i, molecule in enumerate(molecules):
    img_path = f"molecule_{i}.png"
    Draw.MolToImageFile(molecule, img_path)
    image_paths.append(img_path)

# Number of molecules per row
mols_per_row = 5

# Calculate the number of rows needed
num_rows = len(smiles) // mols_per_row
if len(smiles) % mols_per_row:
    num_rows += 1

# Create a Matplotlib figure to display images in a grid
plt.figure(figsize=(mols_per_row * 5, num_rows * 5))

# Add a title with custom styling
plt.suptitle("Random Samples", fontsize=16, fontweight="bold")

for i, img_path in enumerate(image_paths):
    img = mpimg.imread(img_path)
    ax = plt.subplot(num_rows, mols_per_row, i + 1)

    # Remove axes
    ax.axis("off")

    plt.imshow(img)

# Adjust layout to avoid overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("random_samples_grid.png")
plt.show()
