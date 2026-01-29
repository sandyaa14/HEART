actor_ids = list(range(1, 25))  # Actors 01–24

# Filename format:
# MM-CC-VV-VL-EE-II-AA.wav
# Example: 03-01-06-01-02-01-12.wav

modalities = ["03"]              # Audio only
channels = ["01"]                # Speech
vocals = ["01", "02"]            # Vocal channel
emotions = ["01","02","03","04","05","06","07","08"]  # 8 emotions
intensities = ["01","02"]        # normal, strong
statements = ["01","02"]         # two statements
repetitions = ["01","02"]        # two repetitions

filenames = []

for A in actor_ids:
    actor = f"{A:02d}"
    for V in vocals:
        for E in emotions:
            for I in intensities:
                for S in statements:
                    for R in repetitions:
                        fname = f"03-01-{E}-{I}-{S}-{R}-{actor}.wav"
                        filenames.append(fname)

# Sort filenames to match standard RAVDESS ordering
filenames_sorted = sorted(filenames)

# Save to file
with open("filenames_sorted.txt", "w") as f:
    for name in filenames_sorted:
        f.write(name + "\n")

print("Generated", len(filenames_sorted), "filenames")
print("Saved to filenames_sorted.txt")
