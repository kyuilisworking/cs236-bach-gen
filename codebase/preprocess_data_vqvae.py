import pretty_midi
import numpy as np
import torch
import pickle
import os


def create_preprocessed_dataset(
    data_dir_path,
    output_data_path,
    reconstructed_midi_dir_path,
    # seq_len=64,  # in quarter notes
    override=False,
):
    print(f"Creating a preprocessed dataset for VQVAE from {data_dir_path}")
    # print(f"Creating a preprocessed dataset of sequence length {seq_len}")
    # Check if the output file exists and handle override option
    if os.path.exists(output_data_path):
        if override:
            print(f"Cleaning up existing file at {output_data_path}")
            os.remove(output_data_path)
        else:
            print("Error: Output file already exists. Use override=True to overwrite.")
            return

    all_sequences = []
    print(f"Going through all files in directory {data_dir_path}...")
    csv_saved = False

    for filename in os.listdir(data_dir_path):
        seq_idx = 0
        print(f"Processing file: {filename}...")

        if filename.endswith(".mid") or filename.endswith(".midi"):
            midi_file_path = os.path.join(data_dir_path, filename)
            multihot_sequences = midi_to_multihot(midi_file_path)

            if multihot_sequences is None or multihot_sequences.shape[0] == 0:
                print(f"Processing failed for file: {filename}. Moving on.")
                continue

            for sequence in multihot_sequences:
                if not contains_long_pause(sequence):
                    all_sequences.append(sequence)
                    reconstruct_midi_from_multihot_seq(
                        sequence,
                        get_reconstruction_path(
                            reconstruction_dir=reconstructed_midi_dir_path,
                            original_filename=filename,
                            seq_idx=seq_idx,
                        ),
                    )
                    seq_idx += 1

    combined_sequences = np.array(all_sequences)
    combined_sequences = combined_sequences.reshape(-1, 64, 4, 88)

    with open(output_data_path, "wb") as f:
        pickle.dump(combined_sequences, f)
        print(f"Dataset created and saved to {output_data_path}")


def midi_to_multihot(filepath):
    # Load the MIDI file
    try:
        midi_data = pretty_midi.PrettyMIDI(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None  # Return an empty list if there's an error loading the file

    # 1. Get tempo change times and augment with start and end times
    tempo_changes = midi_data.get_tempo_changes()
    last_note_end_time = max(note.end for note in midi_data.instruments[0].notes)
    split_points = [0] + list(tempo_changes[1]) + [last_note_end_time]

    all_sequences = []

    # Iterate over each tempo section
    for i in range(len(split_points) - 1):
        start_time, end_time = split_points[i], split_points[i + 1]

        # 2. Calculate unit_duration
        note_durations = {}
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                if start_time <= note.start < end_time:
                    duration = round(note.end - note.start, 3)
                    note_durations[duration] = note_durations.get(duration, 0) + 1

        total_notes = sum(note_durations.values())

        if total_notes == 0:
            continue

        try:
            unit_duration = min(
                duration
                for duration, count in note_durations.items()
                if count / total_notes > 0.1
            )
        except Exception as e:
            print(note_durations)
            return None

        # 3. Generate multi-hot vectors
        note_sequence = []
        current_time = start_time
        while current_time < end_time:
            vector = np.zeros(88)
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    if current_time <= note.start < current_time + unit_duration:
                        vector[note.pitch - 21] = 1  # Assuming piano range
            note_sequence.append(vector)
            current_time += unit_duration

        # 4. Remove leading and trailing pauses
        # New Step: Trim leading and trailing pauses
        first_non_pause = next(
            (i for i, vec in enumerate(note_sequence) if np.any(vec)), None
        )
        last_non_pause = next(
            (i for i, vec in enumerate(reversed(note_sequence)) if np.any(vec)), None
        )

        if first_non_pause is not None and last_non_pause is not None:
            note_sequence = note_sequence[
                first_non_pause : len(note_sequence) - last_non_pause
            ]
        elif first_non_pause is None:  # Entire sequence is a pause
            note_sequence = []

        # 5. Reshape note_sequence
        sequence_length = len(note_sequence)
        sequence_length -= sequence_length % 256  # Remove leftovers
        reshaped_sequence = np.reshape(note_sequence[:sequence_length], (-1, 256, 88))
        all_sequences.append(reshaped_sequence)

    if len(all_sequences) == 0:
        return None

    # 6. Stack sequences
    all_sequences = np.concatenate(all_sequences, axis=0)

    # 7. Final reshaping
    final_shape = all_sequences.shape[0], 256, 88
    final_data = np.reshape(all_sequences, final_shape)

    return final_data


def contains_long_pause(sequence, max_consec_pause=4):
    """
    Check if the given sequence contains a long pause.
    :param sequence: A numpy array representing the sequence of multi-hot encoded notes.
    :param max_consec_pause: Maximum allowed consecutive timesteps with no note played.
    :return: True if there's a long pause, False otherwise.
    """
    consec_pause_count = 0

    for timestep in sequence:
        # Check if all notes in this timestep are not played (i.e., all zeros)
        if np.all(timestep == 0):
            consec_pause_count += 1
            # If the count exceeds max_consec_pause, return True
            if consec_pause_count > max_consec_pause:
                return True
        else:
            # Reset the consecutive count if a note is played
            consec_pause_count = 0

    return False


def reconstruct_midi_from_multihot_seq(
    seq, output_midi_path, sixteenth_note_duration=0.3
):
    """
    Reconstruct a MIDI file from a multi-hot encoded sequence.

    :param seq: A numpy array of shape [seq_len, 88] representing the multi-hot encoded sequence.
    :param output_midi_path: Path where the reconstructed MIDI file will be saved.
    :param sixteenth_note_duration: Duration of a sixteenth note in seconds.
    """
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano = pretty_midi.Instrument(program=piano_program)

    for time_step in range(seq.shape[0]):  # For each timestep (256)
        for note in range(seq.shape[1]):  # For each note in the timestep (88)
            if seq[time_step, note] == 1:
                # Calculate the start and end times of the note
                start_time = time_step * sixteenth_note_duration
                end_time = start_time + sixteenth_note_duration

                # Create a Note object and add it to the piano instrument
                midi_note = pretty_midi.Note(
                    velocity=100,  # Adjust the velocity if needed
                    pitch=note + 21,  # Offset by 21 to match the MIDI pitch
                    start=start_time,
                    end=end_time,
                )
                piano.notes.append(midi_note)

    # Add the piano instrument to the PrettyMIDI object
    midi.instruments.append(piano)

    # Write out the MIDI data to the output file
    midi.write(output_midi_path)


def get_reconstruction_path(reconstruction_dir, original_filename, seq_idx):
    reconstruction_path = os.path.join(
        reconstruction_dir,
        os.path.splitext(original_filename)[0] + f"{seq_idx}.mid",
    )

    return reconstruction_path


if __name__ == "__main__":
    data_dir_path = "../data/midi_files"
    output_data_path = "../data/training_data/polyphonic_data_vqvae.pkl"
    reconstructed_midi_dir_path = "../data/training_data/reconstructed_midi_vqvae"

    # Call the function with the paths
    create_preprocessed_dataset(
        data_dir_path, output_data_path, reconstructed_midi_dir_path, override=True
    )
