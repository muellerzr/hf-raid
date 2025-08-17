from fastcore.xtras import Path
from fastcore.foundation import L, Self

def split_folder(base_location: Path, *destinations:Path):
    """
    Takes folder located at `base_location` and splits its
    contents evenly to `destinations`. Specifically will resolve
    in the following order:
    1.  Files that are not `.safetensors` will go into the first
        location in `destinations`
    2.  `.safetensor` files are then split evenly between 
        `destinations`
    """
    base_location = Path(base_location)
    destinations = L(Path(d) for d in destinations)
    all_files = L(base_location.ls())
    st = base_location.ls(file_exts=".safetensors")
    non_st = all_files.filter(lambda x: x not in st)
    destinations.map(Self.mkdir(exist_ok=True, parents=True))

    non_st.map(lambda x: x.rename(destinations[0]/x.name))

    st = sorted(st) # Ensure files are in order
    n_dest = len(destinations)
    chunk_size = len(st) // n_dest
    remainder = len(st) % n_dest
    
    start = 0
    for i, dest in enumerate(destinations):
        # Add one extra file for each destination until remainder is used up
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        
        # Move files for this destination
        L(st[start:end]).map(lambda x: x.rename(dest/x.name))
            
        start = end
    
