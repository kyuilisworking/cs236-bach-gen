import pretty_midi
import numpy as np
import pickle
import os
import pandas as pd


def create_preprocessed_dataset(
    data_dir_path,
    output_data_path,
    reconstructed_midi_dir_path,
    seq_len=64,  # in quarter notes
    override=False,
):
    print(f"Creating a preprocessed dataset of sequence length {seq_len}")
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
            multihot_sequence = midi_to_multihot(midi_file_path)

            midi_file_path = os.path.join(data_dir_path, filename)
            multihot_sequence = midi_to_multihot(midi_file_path)

            if multihot_sequence is None:
                print(f"Processing failed for file: {filename}. Moving on.")
                continue

            # Calculate the number of sixteenth notes per sequence
            steps_per_sequence = seq_len * 4

            # Subdivide the sequence

            for start in range(0, multihot_sequence.shape[0], steps_per_sequence):
                end = start + steps_per_sequence
                if end <= multihot_sequence.shape[0]:
                    sub_sequence = multihot_sequence[start:end]
                    if not contains_long_pause(sub_sequence, max_consec_pause=4):
                        reconstruction_path = os.path.join(
                            reconstructed_midi_dir_path,
                            os.path.splitext(filename)[0] + f"{seq_idx}.mid",
                        )
                        reconstruct_midi_from_multihot_seq(
                            sub_sequence, reconstruction_path
                        )
                        if not csv_saved:
                            csv_path = os.path.join(
                                reconstructed_midi_dir_path,
                                os.path.splitext(filename)[0] + f"{seq_idx}.csv",
                            )
                            save_csv(sub_sequence, csv_path)
                            csv_saved = True
                        all_sequences.append(sub_sequence.reshape(seq_len, 4, 88))
                        seq_idx += 1
    combined_sequences = np.array(all_sequences)
    with open(output_data_path, "wb") as f:
        pickle.dump(combined_sequences, f)
        print(f"Dataset created and saved to {output_data_path}")


def midi_to_multihot(filepath, quantization=4):
    """
    Converts a MIDI file to a sequence of multi-hot vectors.
    :param filepath: Path to the MIDI file.
    :param quantization: Number of steps per quarter note. Default is 4 for sixteenth notes.
    :return: A numpy array of multi-hot vectors.
    """

    # Load MIDI file
    try:
        midi_data = pretty_midi.PrettyMIDI(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None  # Return an empty list if there's an error loading the file

    # Get the length of the MIDI file in sixteenth notes
    tempo = midi_data.estimate_tempo()

    end_time = midi_data.get_end_time()
    total_steps = int(end_time * midi_data.resolution / quantization)

    # Create an empty array for multi-hot encoding (88 keys)
    multihot_sequence = np.zeros((total_steps, 88), dtype=np.int8)

    # Process notes
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Adjust the MIDI note number to fit the 88-key piano range
            adjusted_note = max(min(note.pitch - 21, 87), 0)

            # Calculate the start and end steps
            start_step = int(note.start * midi_data.resolution / quantization)
            end_step = int(note.end * midi_data.resolution / quantization)

            # Set the corresponding steps to 1
            multihot_sequence[start_step:end_step, adjusted_note] = 1

    return multihot_sequence


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

    :param seq: A numpy array of shape [256, 88] representing the multi-hot encoded sequence.
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


def save_csv(sequence, output_path):
    np.savetxt(output_path, sequence, delimiter=",", fmt="%d")


if __name__ == "__main__":
    data_dir_path = "../data/midi_files"
    output_data_path = "../data/training_data/polyphonic_data_vqvae.pkl"
    reconstructed_midi_dir_path = "../data/training_data/reconstructed_midi_vqvae"

    # Call the function with the paths
    create_preprocessed_dataset(
        data_dir_path, output_data_path, reconstructed_midi_dir_path, override=True
    )
