import subprocess
import sys
import time
from typing import Optional

from IPython.display import Image, display


# Initialize and check the installation of OgreInterface
def initialize_Ogre(check_install):
    if check_install:
        try:
            # Check if OgreInterface is installed
            result = subprocess.run([sys.executable, "-m", "pip", "show", "OgreInterface"], capture_output=True,
                                    text=True)

            if result.returncode != 0:
                # OgreInterface is not installed, install it
                print("OgreInterface is not installed. Installing now...")
                subprocess.run([sys.executable, "-m", "pip", "install", "OgreInterface"], check=True)
                print("OgreInterface has been installed successfully.")
            else:
                # OgreInterface is installed, check for updates
                print("OgreInterface is installed. Checking for updates...")
                subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "OgreInterface"], check=True)
                print("OgreInterface is up to date.")

        except subprocess.CalledProcessError as e:
            # Handle any errors that occur during installation or update
            print(f"An error occurred while trying to install/update OgreInterface: {e}")
            sys.exit(1)

    # Import the necessary modules from OgreInterface
    global IonicInterfaceSearch, MLIPInterfaceSearch, MillerSearch, InterfaceGenerator, SurfaceGenerator
    from OgreInterface.workflows.interface_search import IonicInterfaceSearch, MLIPInterfaceSearch
    from OgreInterface.miller import MillerSearch
    from OgreInterface.generate import InterfaceGenerator, SurfaceGenerator
    print("OgreInterface imported successfully.")


# Function for lattice matching between two materials
def lattice_matching(CIF_folder, material_A, material_B, max_hkl_A, max_hkl_B, max_strain, max_area, image_dpi,
                     image_scale,
                     sort_by_area_first=True):
    start_time = time.time()
    sub_path = CIF_folder + material_A + ".cif"  # Path to material A CIF file
    film_path = CIF_folder + material_B + ".cif"  # Path to material B CIF file

    # Initialize the MillerSearch with or without specified max_area depending on the user's choice  
    if isinstance(max_area, (int, float)):
        ms = MillerSearch(
            substrate=sub_path,
            film=film_path,
            max_substrate_index=max_hkl_A,
            max_film_index=max_hkl_B,
            max_strain=max_strain,
            max_area=max_area,
            refine_structure=False,
            sort_by_area_first=sort_by_area_first,
        )
    else:
        ms = MillerSearch(
            substrate=sub_path,
            film=film_path,
            max_substrate_index=max_hkl_A,
            max_film_index=max_hkl_B,
            max_strain=max_strain,
            refine_structure=False,
            sort_by_area_first=sort_by_area_first,
        )

    ms.run_scan()  # Run the MillerSearch scan
    misfit_fig_name = f"misfit_plot_{material_A}_{material_B}_max_{max_hkl_A}_{max_hkl_B}_prioritizing_{'area' if sort_by_area_first else 'strain'}.png"
    ms.plot_misfits(dpi=image_dpi, fontsize=14, labelrotation=-45, figure_scale=1,
                    output=misfit_fig_name)  # Plot the misfit results
    print("Elapsed Time =", round(time.time() - start_time, 2), 's')  # Display elapsed time
    display(Image(filename=misfit_fig_name, width=image_scale * 500))  # Display the plot image


# Function to visualize supercells of the interface between two materials
def visualize_supercells(CIF_folder, material_A, material_B, hkl_A, hkl_B, mode, image_dpi, max_strain, max_area,
                         image_scale,
                         sort_by_area_first=True):
    start_time = time.time()
    sub_path = CIF_folder + material_A + ".cif"  # Path to material A CIF file
    film_path = CIF_folder + material_B + ".cif"  # Path to material B CIF file

    # Generate the substrate and film surfaces
    subs = SurfaceGenerator.from_file(sub_path, miller_index=hkl_A, layers=1, vacuum=10, refine_structure=False)
    films = SurfaceGenerator.from_file(film_path, miller_index=hkl_B, layers=1, vacuum=10, refine_structure=False)

    sub = subs[0]
    film = films[0]

    # Initialize the InterfaceGenerator with or without max_area depending on its type
    if isinstance(max_area, (int, float)):
        interface_generator = InterfaceGenerator(substrate=sub, film=film, max_strain=max_strain, max_area=max_area,
                                                 interfacial_distance=None, vacuum=60, center=True,
                                                 sort_by_area_first=sort_by_area_first)
    else:
        interface_generator = InterfaceGenerator(substrate=sub, film=film, max_strain=max_strain,
                                                 interfacial_distance=None, vacuum=60, center=True,
                                                 sort_by_area_first=sort_by_area_first)

    interfaces = interface_generator.generate_interfaces()  # Generate the interfaces

    interface_view_name = f"interface_view_{material_A}_{material_B}_hkl_{hkl_A}_{hkl_B}.png"

    # Handle different modes for visualizing the supercells
    if mode == "view_best":
        # View the best interface
        interface = interfaces[0]
        print(interface)
        interface.plot_interface(dpi=image_dpi, display_results=True, output=interface_view_name)
        display(Image(filename=interface_view_name, width=image_scale * 500))

    elif mode == "view_all":
        # View all interfaces (might be a lot!)
        i = 0
        for interface in interfaces:
            print("INTERFACE INDEX = ", i)
            print(interface)
            interface.plot_interface(dpi=image_dpi, display_results=True, output=interface_view_name)
            display(Image(filename=interface_view_name, width=image_scale * 500))
            i = i + 1

    elif isinstance(mode, (int)):
        # View a specified number of interfaces
        number = min(mode, len(interfaces))
        for i in range(number):
            print("INTERFACE INDEX = ", i)
            interface = interfaces[i]
            print(interface)
            interface.plot_interface(dpi=image_dpi, display_results=True, output=interface_view_name)
            display(Image(filename=interface_view_name, width=image_scale * 500))

    else:
        print("Please select a mode: view_best, view_all, or specify the number of supercells to see.")

    print("Elapsed Time =", round(time.time() - start_time, 2), 's')


# Function to perform surface matching between two materials using IonicInterfaceSearch
def surface_matching(CIF_folder, material_A, material_B, hkl_A, hkl_B, label, max_strain, max_area, strain_fraction,
                     supercell_choice, n_particles_PSO,
                     mlip: Optional[str] = None,
                     model_name=None,
                     device="cuda",
                     substrate_layer_grouping_tolerance=None,
                     film_layer_grouping_tolerance=None,
                     filter_on_charge=True,
                     sort_by_area_first=True
                     ):
    start_time = time.time()
    sub_path = CIF_folder + material_A + ".cif"  # Path to material A CIF file
    film_path = CIF_folder + material_B + ".cif"  # Path to material B CIF file

    # Define the output folder name based on the materials and indices
    output_folder_name = material_A + "_" + ''.join(str(x) for x in hkl_A) + "_" + material_B + "_" + ''.join(
        str(x) for x in hkl_B) + label

    # Initialize IonicInterfaceSearch with the given parameters
    if mlip is None:
        iface_search = IonicInterfaceSearch(
            substrate_bulk=sub_path,
            film_bulk=film_path,
            substrate_miller_index=hkl_A,
            film_miller_index=hkl_B,
            max_strain=max_strain,
            max_area=max_area if isinstance(max_area, (int, float)) else None,
            minimum_slab_thickness=18,  # Cutoff of the electrostatic potential (Angstrom)
            # Excludes fragments with high energy (faster, but might miss plausible models)
            use_most_stable_substrate=False,
            # Distribution of strain between substrate and epilayer (1 = substrate strin only)
            substrate_strain_fraction=strain_fraction,
            n_particles_PSO=n_particles_PSO,  # Particles in the swarm optimization (more = slower but more accurate)
            interface_index=supercell_choice,
            refine_film=False,
            refine_substrate=False,
            substrate_layer_grouping_tolerance=substrate_layer_grouping_tolerance,
            film_layer_grouping_tolerance=film_layer_grouping_tolerance,
            filter_on_charge=filter_on_charge,  # Excludes Interfaces of the [+/+] and [-/-] type
            sort_by_area_first=sort_by_area_first,
        )

    else:
        iface_search = MLIPInterfaceSearch(
            substrate_bulk=sub_path,
            film_bulk=film_path,
            substrate_miller_index=hkl_A,
            film_miller_index=hkl_B,
            max_strain=max_strain,
            max_area=max_area if isinstance(max_area, (int, float)) else None,
            minimum_slab_thickness=18,  # Cutoff of the electrostatic potential (Angstrom)
            # Excludes fragments with high energy (faster, but might miss plausible models)
            use_most_stable_substrate=False,
            # Distribution of strain between substrate and epilayer (1 = substrate strin only)
            substrate_strain_fraction=strain_fraction,
            n_particles_PSO=n_particles_PSO,  # Particles in the swarm optimization (more = slower but more accurate)
            interface_index=supercell_choice,
            refine_film=False,
            refine_substrate=False,
            substrate_layer_grouping_tolerance=substrate_layer_grouping_tolerance,
            film_layer_grouping_tolerance=film_layer_grouping_tolerance,
            filter_on_charge=filter_on_charge,  # Excludes Interfaces of the [+/+] and [-/-] type
            sort_by_area_first=sort_by_area_first,
            mlip=mlip,
            model_name=model_name,
            device=device,
        )

    print("Simulations running!")
    # Run the interface search
    iface_search.run_interface_search(
        output_folder=output_folder_name,
    )

    print("Simulations completed! Elapsed Time =", round(time.time() - start_time, 2), 's')


# Main execution block    
if __name__ == "__main__":
    # Initialize Ogre with check_install set to True
    initialize_Ogre(check_install=True)

    # Default arguments for Lattice Matching
    CIF_folder = "All CIFs/"
    material_A = "CsPbBr3_cubic"
    material_B = "Pb4S3Br2"
    max_hkl_A = 2
    max_hkl_B = 1
    max_strain = 0.1
    max_area = "default"
    image_dpi = 300
    image_scale = 1

    # Default arguments for Visualize Supercells
    hkl_A = [1, 0, 0]
    hkl_B = [0, 1, 0]
    mode = "view_best"

    # Default arguments for Surface Matching
    strain_fraction = 0.5
    supercell_choice = 0
    n_particles_PSO = 30
    label = ""

    # Execute the lattice matching process
    lattice_matching(CIF_folder, material_A, material_B, max_hkl_A, max_hkl_B, max_strain, max_area, image_dpi,
                     image_scale)

    # Execute the supercell visualization process
    visualize_supercells(CIF_folder, material_A, material_B, hkl_A, hkl_B, mode, image_dpi, max_strain, max_area,
                         image_scale)

    # Execute the surface matching process
    surface_matching(CIF_folder, material_A, material_B, hkl_A, hkl_B, label, max_strain, max_area, strain_fraction,
                     supercell_choice, n_particles_PSO)
