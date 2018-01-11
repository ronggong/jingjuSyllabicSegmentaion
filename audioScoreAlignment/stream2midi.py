import music21
import subprocess

def stream2midi(music21stream, midi_filename):
    """
    convert from music21 stream to midi
    :param music21stream:
    :param midi_filename:
    :return:
    """
    mf = music21.midi.translate.streamToMidiFile(music21stream)
    mf.open(midi_filename, 'wb')
    mf.write()
    mf.close()

def midi2wav(midi_filename, wav_filename):
    # turn off delay
    # turn off reverb
    # turn off chrous
    # use voice Oohs as the tone bank
    subprocess.call(['timidity', midi_filename, '-OwM', '-o', wav_filename, '-R', '0', '-m', '0', '-EFdelay=d', '-EFreverb=d', '-EFchorus=d',  '-x', 'bank 0\n0 Tone_000/053_Voice_Oohs.pat'])