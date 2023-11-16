import os
import pretty_midi
import pickle


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
                print("Processing failed for file: {filename}. Moving on.")
                continue
            sequences = preprocess_for_ml(vectorized_tracks, sequence_len)
            all_sequences.extend(sequences)
            print(len(all_sequences))

            # Reconstruct and save the MIDI file to validate that the preprocessing works
            reconstructed_midi_path = os.path.join(
                reconstructed_midi_dir_path, os.path.splitext(filename)[0] + ".mid"
            )
            reconstruct_midi_from_vectors(
                vectorized_tracks, reconstructed_midi_path, sixteenth_note_duration
            )

    # Write the processed data to the output file
    with open(preprocessed_output_data_path, "wb") as file:
        pickle.dump(all_sequences, file)

    print("Preprocessing complete. Data saved to", preprocessed_output_data_path)


def midi_to_vectors(midi_file_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    except Exception as e:
        print(f"Error loading {midi_file_path}: {e}")
        return [], None  # Return an empty list if there's an error loading the file

    # Check the time signature
    for time_signature in midi_data.time_signature_changes:
        if time_signature.numerator not in [2, 4] or time_signature.denominator != 4:
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
    for time_signature in midi_data.time_signature_changes:
        if time_signature.numerator not in [2, 4] or time_signature.denominator != 4:
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


def reconstruct_midi_from_vectors_with_note_off(
    vectorized_tracks, output_midi_path, sixteenth_note_duration
):
    print("hello!")
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
            note_on_indices = [i for i, x in enumerate(vector[:129]) if x == 1]
            print(note_on_indices)
            print(vector)
            rest = note_on_indices[128] == 1  # Check if note off event is indicated

            for note_num in note_on_indices:
                if note_num != last_note or rest:
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
                    if not rest:
                        start_time = current_time
                        last_note = note_num
                    else:
                        last_note = None
                continue

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


# create_preprocessed_dataset('../data/midi_files', '../data/training_data/32_note_sequences.pkl', '../data/training_data/reconstructed_midi', sequence_len=32, override=False)
