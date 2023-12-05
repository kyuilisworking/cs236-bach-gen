import os
import pretty_midi
import pickle
import pandas as pd


# Example usage
# create_preprocessed_dataset('path_to_data_directory', 'path_to_output_file.pkl', override=True)
def create_preprocessed_dataset(
    data_dir_path,
    preprocessed_output_data_path,
    reconstructed_midi_dir_path,
    sequence_len=32,
    override=False,
):
    print(f"Creating a preprocessed dataset of sequence length {sequence_len}")
    # Check if the output file exists and handle override option
    if os.path.exists(preprocessed_output_data_path):
        if override:
            print(f"Cleaning up existing file at {preprocessed_output_data_path}")
            os.remove(preprocessed_output_data_path)
        else:
            print("Error: Output file already exists. Use override=True to overwrite.")
            return

    all_sequences = []

    print(f"Going through all files in directory {data_dir_path}...")
    # Iterate through all MIDI files in the directory
    for filename in os.listdir(data_dir_path):
        print(f"Processing file: {filename}...")
        if filename.endswith(".mid") or filename.endswith(".midi"):
            midi_file_path = os.path.join(data_dir_path, filename)

            # Process each MIDI file
            vectorized_tracks, sixteenth_note_duration = midi_to_vectors(midi_file_path)
            if len(vectorized_tracks) == 0:
                print(f"Processing failed for file: {filename}. Moving on.")
                continue
            sequences = preprocess_for_ml(vectorized_tracks, sequence_len)
            sequences = filter_out_bad_data(
                sequences=sequences,
                max_consec_repeats=4,
                max_eigth_note_count=2,
                max_quarter_note_count=1,
            )
            print(len(sequences))
            sequences = get_transpositions(sequences=sequences)
            print(len(sequences))
            all_sequences.extend(sequences)
            print(len(all_sequences))
            print(sixteenth_note_duration)
            # if len(all_sequences) > 0:
            #     break

            # # Reconstruct and save the MIDI file to validate that the preprocessing works
            # reconstructed_midi_path = os.path.join(
            #     reconstructed_midi_dir_path, os.path.splitext(filename)[0] + ".mid"
            # )
            # reconstruct_midi_from_vectors(
            #     vectorized_tracks, reconstructed_midi_path, sixteenth_note_duration
            # )

    # Write the processed data to the output file
    with open(preprocessed_output_data_path, "wb") as file:
        pickle.dump(all_sequences, file)

    # reconstructed_midi_path = os.path.join(
    #     reconstructed_midi_dir_path, os.path.splitext(filename)[0] + ".mid"
    # )

    # save the CSV
    base_name = os.path.splitext(preprocessed_output_data_path)[0]
    csv_path_name = f"{base_name}.csv"

    convert_sequences_to_csv(all_sequences, csv_path_name)

    i = 1
    for sequence in all_sequences:
        reconstructed_midi_path = os.path.join(reconstructed_midi_dir_path, f"{i}.mid")
        i += 1
        reconstruct_midi_from_vectors([sequence], reconstructed_midi_path, 0.1)

    print("Preprocessing complete. Data saved to", preprocessed_output_data_path)


def midi_to_vectors(midi_file_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    except Exception as e:
        print(f"Error loading {midi_file_path}: {e}")
        return [], None  # Return an empty list if there's an error loading the file

    # Check the time signature
    for time_signature in midi_data.time_signature_changes:
        if time_signature.numerator not in [2, 4] or time_signature.denominator not in [
            2,
            4,
        ]:
            return [], None

    # Calculate the duration of a 16th note in seconds
    tempo = midi_data.estimate_tempo()
    beat_duration = 60 / tempo  # Duration of a quarter note
    sixteenth_note_duration = beat_duration / 4

    tracks_vectors = []

    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            current_time = 0
            last_note = None
            last_note_end_time = 0
            track_vectors = []

            while current_time < midi_data.get_end_time():
                vector = [0] * 129
                notes_in_interval = [
                    note
                    for note in instrument.notes
                    if current_time
                    <= note.start
                    < current_time + sixteenth_note_duration
                ]

                if notes_in_interval:
                    # This means a new note is played
                    earliest_note = min(notes_in_interval, key=lambda note: note.start)
                    vector[earliest_note.pitch] = 1

                    last_note = earliest_note.pitch
                    last_note_end_time = earliest_note.end
                elif last_note is not None and current_time < last_note_end_time:
                    # Prolonged note
                    vector[last_note] = 1
                else:
                    # Rest
                    vector[128] = 1
                    last_note = None

                track_vectors.append(vector)
                current_time += sixteenth_note_duration

            tracks_vectors.append(track_vectors)
        break  # this is to only keep the right hands

    return tracks_vectors, sixteenth_note_duration


# creates a 130-length vector for each 16th note
# 128 note-on, note-off, rest
def midi_to_vectors_note_off(midi_file_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    except Exception as e:
        print(f"Error loading {midi_file_path}: {e}")
        return [], None  # Return an empty list if there's an error loading the file

    # Check the time signature
    # for time_signature in midi_data.time_signature_changes:
    #     if time_signature.numerator not in [2, 4] or time_signature.denominator != 4:
    #         return [], None

    # Calculate the duration of a 16th note in seconds
    tempo = midi_data.estimate_tempo()
    beat_duration = 60 / tempo  # Duration of a quarter note
    sixteenth_note_duration = beat_duration / 4

    tracks_vectors = []

    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            current_time = 0
            last_note = None
            last_note_end_time = 0
            track_vectors = []

            while current_time < midi_data.get_end_time():
                vector = [0] * 130
                notes_in_interval = [
                    note
                    for note in instrument.notes
                    if current_time
                    <= note.start
                    < current_time + sixteenth_note_duration
                ]

                if notes_in_interval:
                    # This means a new note is played
                    earliest_note = min(notes_in_interval, key=lambda note: note.start)
                    vector[earliest_note.pitch] = 1

                    # Note off for repeated note
                    if (
                        last_note == earliest_note.pitch
                        and current_time >= last_note_end_time
                    ):
                        vector[128] = 1

                    last_note = earliest_note.pitch
                    last_note_end_time = earliest_note.end
                elif last_note is not None and current_time < last_note_end_time:
                    # Prolonged note
                    vector[last_note] = 1
                else:
                    # Rest
                    vector[129] = 1
                    last_note = None

                track_vectors.append(vector)
                current_time += sixteenth_note_duration

            tracks_vectors.append(track_vectors)

    return tracks_vectors, sixteenth_note_duration


# # Example usage
# # tracks = midi_to_vectors('path_to_your_midi_file.mid')
# def midi_to_vectors(midi_file_path):
#     try:
#         midi_data = pretty_midi.PrettyMIDI(midi_file_path)
#     except Exception as e:
#         print(f"Error loading {midi_file_path}: {e}")
#         return []

#     # Check the time signature
#     for time_signature in midi_data.time_signature_changes:
#         if time_signature.numerator not in [2, 4] or time_signature.denominator not in [2, 4]:
#             # If time signature is not simple (2/4, 2/2, or 4/4), return an empty list
#             return []

#     # Calculate the duration of a 16th note in seconds
#     tempo = midi_data.estimate_tempo()
#     beat_duration = 60 / tempo  # Duration of a quarter note
#     sixteenth_note_duration = beat_duration / 4

#     tracks_vectors = []

#     for instrument in midi_data.instruments:
#         if not instrument.is_drum:
#             current_time = 0
#             track_vectors = []
#             notes_on = set()

#             while current_time < midi_data.get_end_time():
#                 vector = [0] * 130

#                 # Check for 'Note On' and 'Note Off' events
#                 for note in instrument.notes:
#                     if note.start <= current_time < note.end:
#                         vector[note.pitch] = 1
#                         notes_on.add(note.pitch)
#                     elif note.end <= current_time and note.pitch in notes_on:
#                         vector[128] = 1  # Note Off
#                         notes_on.remove(note.pitch)

#                 # Check for a rest (no notes playing)
#                 if len(notes_on) == 0:
#                     vector[129] = 1  # Rest

#                 track_vectors.append(vector)
#                 current_time += sixteenth_note_duration

#             tracks_vectors.append(track_vectors)

#     return tracks_vectors


# Example usage
# vectorized_tracks = midi_to_vectors('path_to_your_midi_file.mid')
# sequences = preprocess_for_ml(vectorized_tracks, sequence_len)
def preprocess_for_ml(vectorized_tracks, sequence_len):
    processed_sequences = []

    for track_vectors in vectorized_tracks:
        # Break each track's vector data into sequences of length sequence_len
        for i in range(0, len(track_vectors), sequence_len):
            sequence = track_vectors[i : i + sequence_len]
            if len(sequence) == sequence_len:
                processed_sequences.append(sequence)

    return processed_sequences


def filter_out_bad_data(
    sequences, max_consec_repeats=4, max_eigth_note_count=6, max_quarter_note_count=2
):
    filtered_sequences = filter_super_long_notes(
        sequences, max_consec_repeats=max_consec_repeats
    )
    # filtered_sequences = filter_too_many_eigth_or_quarter_notes(
    #     filtered_sequences,
    #     max_eigth_note_count=max_eigth_note_count,
    #     max_quarter_note_count=max_quarter_note_count,
    # )
    return filtered_sequences


def filter_super_long_notes(sequences, max_consec_repeats):
    filtered_sequences = []

    for sequence in sequences:
        current_note = None
        consecutive_count = 0
        should_add = True

        for note in sequence:
            if note == current_note:
                consecutive_count += 1
            else:
                current_note = note
                consecutive_count = 1

            if consecutive_count > max_consec_repeats:
                should_add = False
                break

        if should_add:
            filtered_sequences.append(sequence)

    return filtered_sequences


def filter_too_many_eigth_or_quarter_notes(
    sequences, max_eigth_note_count, max_quarter_note_count
):
    filtered_sequences = []

    for sequence in sequences:
        eigth_notes, quarter_notes = count_notes(sequence)
        if (
            eigth_notes <= max_eigth_note_count
            and quarter_notes < max_quarter_note_count
        ):
            filtered_sequences.append(sequence)

    return filtered_sequences


def get_transpositions(sequences, interval_range=(-6, 6)):
    transposed_data = []
    for interval in range(interval_range[0], interval_range[1] + 1):
        for seq in sequences:
            new_seq = []
            for note in seq:
                if note[128] == 0:  # Check if it's a note
                    original_pitch = note.index(1)  # Find the current pitch
                    new_pitch = original_pitch + interval

                    # Handle octave wrapping or limit to MIDI range
                    new_pitch = max(0, min(new_pitch, 127))

                    # Create a new vector with the transposed note
                    new_note = [0] * 129
                    new_note[new_pitch] = 1
                else:
                    # If it's a rest, keep the vector unchanged
                    new_note = note.copy()

                new_seq.append(new_note)
            transposed_data.append(new_seq)
    return transposed_data


def count_notes(sequence):
    eighth_notes = 0
    quarter_notes = 0

    current_note = None
    consecutive_count = 0

    for note in sequence + [
        [None]
    ]:  # Adding a dummy note at the end to trigger the final check
        if note == current_note:
            consecutive_count += 1
        else:
            # Check for eighth and quarter notes when note changes
            if consecutive_count == 2:
                eighth_notes += 1
            elif consecutive_count == 4:
                quarter_notes += 1

            current_note = note
            consecutive_count = 1

    return eighth_notes, quarter_notes


def reconstruct_midi_from_vectors(
    vectorized_tracks, output_midi_path, sixteenth_note_duration
):
    # Create a PrettyMIDI object
    reconstructed_midi = pretty_midi.PrettyMIDI()
    # Create an instrument instance (assuming a piano instrument)
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    )

    current_time = 0.0
    for track in vectorized_tracks:
        last_note = None
        for vector in track:
            if vector[128] == 1:
                # handle rest
                if last_note is not None:
                    end_time = current_time
                    note = pretty_midi.Note(
                        velocity=100, pitch=last_note, start=start_time, end=end_time
                    )
                    instrument.notes.append(note)
                    last_note = None
                current_time += sixteenth_note_duration
                last_note = None
                continue

            # handle actual note
            note_on_indices = [i for i, x in enumerate(vector[:128]) if x == 1]

            for note_num in note_on_indices:
                if note_num != last_note:
                    # End the last note
                    if last_note is not None:
                        end_time = current_time
                        note = pretty_midi.Note(
                            velocity=100,
                            pitch=last_note,
                            start=start_time,
                            end=end_time,
                        )
                        instrument.notes.append(note)
                    # Start a new note
                    start_time = current_time
                    last_note = note_num

            # Move to the next time step
            current_time += sixteenth_note_duration

        # End the last note of the track
        if last_note is not None:
            end_time = current_time
            note = pretty_midi.Note(
                velocity=100, pitch=last_note, start=start_time, end=end_time
            )
            instrument.notes.append(note)

    reconstructed_midi.instruments.append(instrument)
    reconstructed_midi.write(output_midi_path)


# TODO: FIX
def reconstruct_midi_from_vectors_with_note_off(
    vectorized_tracks, output_midi_path, sixteenth_note_duration
):
    # Create a PrettyMIDI object
    reconstructed_midi = pretty_midi.PrettyMIDI()
    # Create an instrument instance (assuming a piano instrument)
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    )

    current_time = 0.0
    for track in vectorized_tracks:
        last_note = None
        for vector in track:
            note_on_indices = [i for i, x in enumerate(vector[:128]) if x == 1]
            rest = vector[128] == 1  # Check if note off event is indicated

            for note_num in note_on_indices:
                if note_num != last_note or note_off:
                    # End the last note
                    if last_note is not None:
                        end_time = current_time
                        note = pretty_midi.Note(
                            velocity=100,
                            pitch=last_note,
                            start=start_time,
                            end=end_time,
                        )
                        instrument.notes.append(note)
                    # Start a new note
                    start_time = current_time
                    last_note = note_num

            # Move to the next time step
            current_time += sixteenth_note_duration

        # End the last note of the track
        if last_note is not None:
            end_time = current_time
            note = pretty_midi.Note(
                velocity=100, pitch=last_note, start=start_time, end=end_time
            )
            instrument.notes.append(note)

    reconstructed_midi.instruments.append(instrument)
    reconstructed_midi.write(output_midi_path)


def convert_sequences_to_csv(sequences, output_file_name):
    # Function to convert one-hot vector to 1-indexed number
    def one_hot_to_number(vector):
        return vector.index(1) + 1

    # Convert each sequence of one-hot vectors to a list of numbers
    converted_sequences = []
    for sequence in sequences:
        converted_sequence = [one_hot_to_number(note) for note in sequence]
        converted_sequences.append(converted_sequence)

    # Create a DataFrame
    df = pd.DataFrame(converted_sequences)

    # Export to CSV
    df.to_csv(output_file_name, index=False)


# create_preprocessed_dataset(
#     data_dir_path="../data/midi_files",
#     preprocessed_output_data_path="../data/training_data/32_note_sequences.pkl",
#     reconstructed_midi_dir_path="../data/training_data/reconstructed_midi",
#     sequence_len=32,
#     override=False,
# )

# create_preprocessed_dataset(
#     data_dir_path="../data/midi_files",
#     preprocessed_output_data_path="../data/training_data/32_note_sequences_filtered.pkl",
#     reconstructed_midi_dir_path="../data/training_data/reconstructed_midi_filtered",
#     sequence_len=32,
#     override=True,
# )
