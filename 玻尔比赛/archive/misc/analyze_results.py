#!/usr/bin/env python3
"""
Analysis script for basalt fiber Li-S battery separator calculations.
Parses CP2K and LAMMPS outputs to extract:
1. Adsorption energies of polysulfides on oxide surfaces
2. Electronic structure properties
3. MD thermal stability and Li+ diffusion coefficients
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CP2K Output Parser
# ============================================================

def parse_cp2k_energy(log_file):
    """Extract final total energy from CP2K log file"""
    energy = None
    with open(log_file, 'r') as f:
        for line in f:
            # Look for energy in GEO_OPT or SCF output
            if "ENERGY| Total FORCE_EVAL" in line:
                match = re.search(r'FORCE_EVAL \( GEO_OPT \) energy \(a\.u\.\):\s+([-\d.]+)', line)
                if match:
                    energy = float(match.group(1))
            elif "Total energy:" in line:
                match = re.search(r'Total energy:\s+([-\d.]+)', line)
                if match:
                    energy = float(match.group(1))
            elif "ENERGY| Total FORCE_EVAL" in line:
                match = re.search(r'energy \(a\.u\.\):\s+([-\d.]+)', line)
                if match:
                    energy = float(match.group(1))
    return energy

def parse_cp2k_forces(log_file):
    """Extract forces from CP2K trajectory"""
    forces = []
    with open(log_file, 'r') as f:
        content = f.read()
    # Parse force blocks
    force_blocks = re.findall(r'ATOMIC FORCES in.*?\n(.*?)\n\s*SUM OF', content, re.DOTALL)
    for block in force_blocks:
        block_forces = []
        for line in block.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 6:
                try:
                    fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                    block_forces.append([fx, fy, fz])
                except (ValueError, IndexError):
                    continue
        if block_forces:
            forces.append(block_forces)
    return forces

def parse_cp2k_bandgap(log_file):
    """Extract HOMO-LUMO gap from CP2K output"""
    gap = None
    with open(log_file, 'r') as f:
        for line in f:
            if "HOMO - LUMO gap" in line:
                match = re.search(r'gap.*?([-\d.]+)', line)
                if match:
                    gap = float(match.group(1))
    return gap

# ============================================================
# Adsorption Energy Calculator
# ============================================================

def calculate_adsorption_energies(results_dir):
    """
    Calculate adsorption energies:
    E_ads = E(surface+molecule) - E(surface) - E(molecule)
    
    Negative E_ads means favorable adsorption.
    """
    # Conversion: 1 Hartree = 27.211 eV
    Ha_to_eV = 27.2114
    
    surfaces = ["SiO2", "Al2O3", "Fe2O3"]
    molecules = ["Li2S", "Li2S4", "Li2S6"]
    
    energies = {}
    adsorption_energies = {}
    
    # Parse surface energies
    for surf in surfaces:
        log_path = os.path.join(results_dir, f"DFT_surfaces/{surf}_001/log")
        if os.path.exists(log_path):
            e = parse_cp2k_energy(log_path)
            if e is not None:
                energies[f"surface_{surf}"] = e
                print(f"  {surf}(001) surface energy: {e:.6f} Ha ({e*Ha_to_eV:.3f} eV)")
    
    # Parse molecule energies
    for mol in molecules:
        log_path = os.path.join(results_dir, f"DFT_molecules/{mol}/log")
        if os.path.exists(log_path):
            e = parse_cp2k_energy(log_path)
            if e is not None:
                energies[f"mol_{mol}"] = e
                print(f"  {mol} molecule energy: {e:.6f} Ha ({e*Ha_to_eV:.3f} eV)")
    
    # Parse adsorption energies
    for surf in surfaces:
        for mol in molecules:
            key = f"{surf}_{mol}"
            log_path = os.path.join(results_dir, f"DFT_adsorption/{key}/log")
            if os.path.exists(log_path):
                e = parse_cp2k_energy(log_path)
                if e is not None:
                    energies[f"ads_{key}"] = e
                    
                    # Calculate adsorption energy
                    e_surf = energies.get(f"surface_{surf}")
                    e_mol = energies.get(f"mol_{mol}")
                    if e_surf is not None and e_mol is not None:
                        e_ads = (e - e_surf - e_mol) * Ha_to_eV  # in eV
                        adsorption_energies[key] = e_ads
                        print(f"  {key}: E_ads = {e_ads:.3f} eV")
    
    return energies, adsorption_energies

# ============================================================
# MD Analysis
# ============================================================

def parse_msd_file(msd_file):
    """Parse LAMMPS MSD output file"""
    data = np.loadtxt(msd_file, skiprows=1)
    # Columns: step, MSD_x, MSD_y, MSD_z, MSD_total
    return data

def calculate_diffusion_coefficient(msd_data, timestep_ps=0.001):
    """
    Calculate diffusion coefficient from MSD using Einstein relation:
    D = MSD / (6 * t) for 3D diffusion
    
    Args:
        msd_data: array with columns [step, msd_x, msd_y, msd_z, msd_total]
        timestep_ps: timestep in picoseconds
    Returns:
        D in cm²/s
    """
    # Use the linear region (last 50% of data)
    n = len(msd_data)
    start = n // 2
    
    time_ps = msd_data[start:, 0] * timestep_ps
    msd_total = msd_data[start:, 4]  # Å²
    
    # Linear fit: MSD = 6*D*t + c
    if len(time_ps) > 2:
        slope, intercept = np.polyfit(time_ps, msd_total, 1)
        # D = slope / 6 (in Å²/ps)
        D_A2_ps = slope / 6.0
        # Convert to cm²/s: 1 Å²/ps = 1e-4 cm²/s
        D_cm2_s = D_A2_ps * 1e-4
        return D_cm2_s, slope, intercept
    return None, None, None

def parse_lammps_thermo(log_file):
    """Parse LAMMPS thermo output"""
    data = []
    header = None
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith("Step"):
                header = line.split()
            elif header and line.strip() and line[0].isdigit():
                try:
                    values = [float(x) for x in line.split()]
                    data.append(values)
                except ValueError:
                    continue
    if data:
        return np.array(data), header
    return None, None

# ============================================================
# Plotting Functions
# ============================================================

def plot_adsorption_energies(adsorption_energies, output_file="adsorption_energies.png"):
    """Plot adsorption energies as grouped bar chart"""
    surfaces = ["SiO2", "Al2O3", "Fe2O3"]
    molecules = ["Li2S", "Li2S4", "Li2S6"]
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(surfaces))
    width = 0.25
    
    for i, mol in enumerate(molecules):
        values = []
        for surf in surfaces:
            key = f"{surf}_{mol}"
            values.append(adsorption_energies.get(key, 0))
        ax.bar(x + i * width, values, width, label=mol, color=colors[i])
    
    ax.set_xlabel('Surface', fontsize=14)
    ax.set_ylabel('Adsorption Energy (eV)', fontsize=14)
    ax.set_title('Polysulfide Adsorption Energies on Basalt Fiber Components', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(surfaces)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved adsorption energy plot to {output_file}")

def plot_msd(msd_data, output_file="msd_plot.png", timestep_ps=0.001):
    """Plot MSD vs time"""
    time_ns = msd_data[:, 0] * timestep_ps / 1000  # Convert to ns
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_ns, msd_data[:, 4], 'b-', label='Total MSD', linewidth=2)
    ax.plot(time_ns, msd_data[:, 1], 'r--', label='MSD_x', alpha=0.7)
    ax.plot(time_ns, msd_data[:, 2], 'g--', label='MSD_y', alpha=0.7)
    ax.plot(time_ns, msd_data[:, 3], 'm--', label='MSD_z', alpha=0.7)
    
    ax.set_xlabel('Time (ns)', fontsize=14)
    ax.set_ylabel('MSD (Å²)', fontsize=14)
    ax.set_title('Li⁺ Mean Square Displacement in Basalt Fiber', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved MSD plot to {output_file}")

def plot_rdf(rdf_file, output_file="rdf_plot.png"):
    """Plot radial distribution function"""
    data = np.loadtxt(rdf_file, skiprows=1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data[:, 0], data[:, 1], 'b-', linewidth=2)
    ax.set_xlabel('r (Å)', fontsize=14)
    ax.set_ylabel('g(r)', fontsize=14)
    ax.set_title('Li-O Radial Distribution Function', fontsize=14)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 6)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved RDF plot to {output_file}")

# ============================================================
# Main Analysis
# ============================================================

if __name__ == "__main__":
    results_dir = "/share/玻尔比赛"
    
    print("=" * 60)
    print("Basalt Fiber Li-S Battery Separator - Analysis Report")
    print("=" * 60)
    
    print("\n--- Adsorption Energies ---")
    energies, ads_energies = calculate_adsorption_energies(results_dir)
    
    if ads_energies:
        plot_adsorption_energies(ads_energies, 
                                os.path.join(results_dir, "results/adsorption_energies.png"))
        
        # Save results to JSON
        with open(os.path.join(results_dir, "results/adsorption_results.json"), 'w') as f:
            json.dump({
                "total_energies_hartree": {k: v for k, v in energies.items()},
                "adsorption_energies_eV": ads_energies
            }, f, indent=2)
    
    print("\n--- MD Analysis ---")
    msd_file = os.path.join(results_dir, "MD_Li_diffusion/msd_Li.dat")
    if os.path.exists(msd_file):
        msd_data = parse_msd_file(msd_file)
        D, slope, intercept = calculate_diffusion_coefficient(msd_data)
        if D is not None:
            print(f"  Li+ diffusion coefficient: {D:.2e} cm²/s")
            plot_msd(msd_data, os.path.join(results_dir, "results/msd_Li.png"))
    
    rdf_file = os.path.join(results_dir, "MD_Li_diffusion/rdf_LiO.dat")
    if os.path.exists(rdf_file):
        plot_rdf(rdf_file, os.path.join(results_dir, "results/rdf_LiO.png"))
    
    print("\n--- Summary ---")
    print(f"  Total DFT calculations: {len(energies)}")
    print(f"  Adsorption systems computed: {len(ads_energies)}")
    print("=" * 60)
