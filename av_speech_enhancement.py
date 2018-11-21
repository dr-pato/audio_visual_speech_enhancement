import sys
import argparse

from mixed_speech_generator import create_mixed_tracks_data
from audio_features import save_spectrograms
from face_landmarks import save_face_landmarks, show_face_landmarks
from target_binary_mask import save_target_binary_masks
from create_dataset_tfrecords import create_dataset
from training import train, Configuration
from testing import test


def parse_args():
    parser = argparse.ArgumentParser(description="""Audio-visual speech enhancement system.
    Try 'av_speech_enhancement <subcommand> --help' for more information about subcommands.
    A full list of subcommands is shown below (positional arguments).""")
    subparsers = parser.add_subparsers(dest='subparser_name')

    # Mixed-speech generator
    mixed_speech_parser = subparsers.add_parser('mixed_speech_generator', description="""Generate mixed-speech samples.
    For each <speaker_id> in <base_speaker_ids> are randomly chosen <num_samples> wavs from directory <data_dir>/s<speaker_id>/<audio_dir>.
    For each chosen sample are generated <num_mix> mixed-speech samples (<num_mix_speakers> noisy samples are selected from directories
    of <noisy_speaker_ids>. Mixed-speech wavs are saved in <data_dir>/<dest_dir>/<speaker_id>.""")
    mixed_speech_parser.add_argument('-data', '--data_dir', required=True, help='The base pathname of dataset')
    mixed_speech_parser.add_argument('-bs', '--base_speaker_ids', nargs='+', type=int, required=True,
                                     help='Speaker IDs of base wav')
    mixed_speech_parser.add_argument('-ns', '--noisy_speaker_ids', nargs='*', type=int, default=[],
                                     help='Speaker IDs of noisy wavs (if omitted are the same of base speaker IDs)')
    mixed_speech_parser.add_argument('-a', '--audio_dir', required=True,
                                     help='The subdirectory that contains wav files to be processed')
    mixed_speech_parser.add_argument('-d', '--dest_dir', required=True,
                                     help='The directory where mixed-speech wavs are saved')
    mixed_speech_parser.add_argument('-num', '--num_samples', type=int, required=True,
                                     help='Number of randomly chosen base wavs for each speaker')
    mixed_speech_parser.add_argument('-mix', '--num_mix', type=int, required=True,
                                     help='Number of mixed-speech wavs generated for each base wav')
    mixed_speech_parser.add_argument('-ms', '--num_mix_speakers', type=int, choices=[1, 2], required=True,
                                     help='Number of wavs of noisy speakers mixed with base wav')
    
    # Compute audio spectrograms
    audio_preprocessing_parser = subparsers.add_parser('audio_preprocessing', description="""Compute audio spectrograms.
    For each <speaker_id> in <speaker_ids> power-law compressed spectrograms are computed for all wavs in
    <data_dir>/s<speaker_id>/<audio_dir>. The spectrograms are saved in NPY format in <data_dir>/s<speaker_id>/<dest_dir>.""")
    audio_preprocessing_parser.add_argument('-data', '--data_dir', required=True, help='The base pathname of dataset')
    audio_preprocessing_parser.add_argument('-s', '--speaker_ids', nargs='+', type=int, required=True,
                                            help='Speaker IDs of wavs to be processed')
    audio_preprocessing_parser.add_argument('-a', '--audio_dir', required=True,
                                            help='The subdirectory that contains wav files to be processed')
    audio_preprocessing_parser.add_argument('-d', '--dest_dir', required=True,
                                            help='The subdirectory where spectrograms are saved')
    audio_preprocessing_parser.add_argument('-sr', '--sample_rate', type=int, default=16000,
                                            help='Desired sample rate (in Hz)')
    audio_preprocessing_parser.add_argument('-ml', '--max_wav_length', type=int, required=True,
                                            help='Set this value to the maximum length (in samples with desidered sample rate) of single wav .')

    # Compute face landmarks
    video_preprocessing_parser = subparsers.add_parser('video_preprocessing', description="""Compute face landmarks.
    For each <speaker_id> in <speaker_ids> face landmarks are computed for all videos in
    <data_dir>/s<speaker_id>/<video_dir>. The face landmarks are saved in TXT format in <data_dir>/s<speaker_id>/<dest_dir>.""")
    video_preprocessing_parser.add_argument('-data', '--data_dir', required=True, help='The base pathname of dataset')
    video_preprocessing_parser.add_argument('-s', '--speaker_ids', nargs='+', type=int, required=True,
                                            help='Speaker IDs of videos to be processed')
    video_preprocessing_parser.add_argument('-v', '--video_dir', required=True,
                                            help='The subdirectory that contains video files to be processed')
    video_preprocessing_parser.add_argument('-d', '--dest_dir', required=True,
                                            help='The subdirectory where output files are saved')
    video_preprocessing_parser.add_argument('-sp', '--shape_predictor', required=True,
                                            help='Path of the file that contains the parameters of Dlib face landmark extractor')
    video_preprocessing_parser.add_argument('-e', '--ext', required=True, default='mpg',
                                            help='The extension of video files')

    # Show face landmarks
    show_landmarks_parser = subparsers.add_parser('show_face_landmarks', description="""Show face landmarks.
    If TXT file of face landmarks is not provided, the face landmark are computed with face landmark extractor.""")
    show_landmarks_parser.add_argument('-v', '--video', required=True,
                                            help='The video file to be processed')
    show_landmarks_parser.add_argument('--fps', type=float, default=25.0,
                                       help='Video frame rate. If it set to zero, you have to press any key to show the video frame-by-frame')
    group = show_landmarks_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-sp', '--shape_predictor', default='',
                       help='The file that contains the parameters of Dlib face landmark extractor')
    group.add_argument('-fl', '--face_landmarks', default='',
                       help='The face landmark TXT file')
    show_landmarks_parser.add_argument('--full', action='store_const', const=True, default=False,
                                       help='Draw connected lines of face landmarks points')

    # Compute target binary masks
    tbm_parser = subparsers.add_parser('tbm_computation', description="""Compute Target Binary Masks (TBM).
    For each <speaker_id> in <speaker_ids> TBMs are computed for all wavs in
    <data_dir>/s<speaker_id>/<audio_dir>. The spectrograms are saved in NPY format in <data_dir>/s<speaker_id>/<dest_dir>.""")
    tbm_parser.add_argument('-data', '--data_dir', required=True, help='The base pathname of dataset')
    tbm_parser.add_argument('-s', '--speaker_ids', nargs='+', type=int, required=True,
                            help='Speaker IDs of wavs to be processed')
    tbm_parser.add_argument('-a', '--audio_dir', required=True,
                            help='The subdirectory that contains wav files to be processed')
    tbm_parser.add_argument('-d', '--dest_dir', required=True,
                            help='The subdirectory where masks are saved')
    tbm_parser.add_argument('-mf', '--mask_factor', type=float, default=0.6,
                            help='A value that controls the amount of T-F units assigned to target speaker (higher values assign more T-F units to noise/silence)')
    tbm_parser.add_argument('-sr', '--sample_rate', type=int, default=16000,
                            help='Desired sample rate (in Hz)')
    tbm_parser.add_argument('-ml', '--max_wav_length', type=int, required=True,
                            help='Set this value to the maximum length (in samples with desired sample rate) of single wav.')
    tbm_parser.add_argument('-nl', '--num_ltass', type=int, default=1000,
                            help='Number of samples of target speaker used to compute LTASS')

    # Create TFRecords
    tfrecords_parser = subparsers.add_parser('tfrecords_generator', description="""Create TFRecords of dataset.
    <data_dir>/<mix_dir> must have the following structure. There are three directories named TRAINING_SET, VALIDATION_SET and
    TEST_SET. These directories cointains subdirectories s<speaker_id> for each speaker. A directory of a speaker
    contains mixed-speech samples created with <mixed_speech_generator> command along with associated pre-computed
    spectrograms (NPY format). <data_dir>/s<speaker_id>/<audio_dir|video_dir|tbm_dir> contains files associated to each
    mixed-speech samples to create a single TFRecord.
    """)
    tfrecords_parser.add_argument('-data', '--data_dir', required=True, help='The base pathname of dataset')
    tfrecords_parser.add_argument('-n', '--num_speakers', type=int, choices=[2, 3], default=2,
                                  help='Numbers of speakers in the mixture (default: 2)')
    tfrecords_parser.add_argument('-m', '--mode', default='fixed', choices=['fixed', 'var'],
                                  help='"fixed" (default) if wavs have all the same length, "var" otherwise')
    tfrecords_parser.add_argument('-d', '--dest_dir', required=True,
                                  help='The subdirectory where TFRecords are saved')
    tfrecords_parser.add_argument('-b', '--base_audio_dir', required=True,
                                  help='The subdirectory that contains base-speech wavs')
    tfrecords_parser.add_argument('-v', '--video_dir', required=True,
                                  help='The subdirectory that contains pre-computed face landmarks (TXT format)')
    tfrecords_parser.add_argument('-tbm', '--tbm_dir', required=True,
                                  help='The subdirectory that contains pre-computed TBM (NPY format)')
    tfrecords_parser.add_argument('-mix', '--mix_audio_dir', required=True,
                                  help='The subdirectory that contains mixed-speech wavs and pre-computed spectrograms (NPY format)')
    tfrecords_parser.add_argument('--delta', type=int, choices=[0, 1, 2], default=1,
                                  help='Select video features. 0: raw face landmarks; 1: motion vectors of face landmarks (default); 2: motion of motion vectors')
    tfrecords_parser.add_argument('-norm', '--norm_data_dir', required=True,
                                  help='The directory where mean and standard deviation of audio and visual features are saved')
    
    # Train audio-visual speech enhancement model
    training_parser = subparsers.add_parser('training', description="""Train an audio-visual speech enhancement model.""")
    training_parser.add_argument('-data', '--data_dir', required=True, help='The base pathname of dataset')
    training_parser.add_argument('-ts', '--train_set', required=True, help='Subdirectory with TFRecords of training set')
    training_parser.add_argument('-vs', '--val_set', required=True, help='Subdirectory with TFRecords of validation set')
    training_parser.add_argument('-e', '--exp', required=True, help='Experiment identifier')
    training_parser.add_argument('-m', '--mode', default='fixed', choices=['fixed', 'var'],
                                 help='"TFRecord type: fixed" (default) or "var"')
    training_parser.add_argument('-vd', '--video_dim', type=int, default=136,
                                 help='Size of a single video frame (default: 136)')
    training_parser.add_argument('-ad', '--audio_dim', type=int, default=257,
                                 help='Size of a single audio frame (default: 257)')
    training_parser.add_argument('-ns', '--num_audio_samples', type=int,
                                 help='Number of samples of audio wav if <mode> is "fixed" (otherwise it is ignored)')
    training_parser.add_argument('--model', choices=['vl2m', 'vl2m_ref', 'av_concat_mask', 'av_concat_spec'],
                                 help='Model type. "av_concat_spec" is the "av_concat_mask" model w/o masking')
    training_parser.add_argument('-o', '--opt', required=True, choices=['sgd', 'adam', 'momentum'],
                                 help='Training optimizer.')
    training_parser.add_argument('-lr', '--learning_rate', type=float, required=True, help='Initial learning rate')
    training_parser.add_argument('-us', '--updating_step', type=int, default=1000,
                                 help='Frequency (in training steps) of updates of learning rate ("sgd" and "momentum" only)')
    training_parser.add_argument('-lc', '--learning_decay', type=float, default=1.0,
                                 help='Learning rate decay ("sgd" and "momentum" only)')
    training_parser.add_argument('-bs', '--batch_size', type=int, help='Training batch size')
    training_parser.add_argument('-ep', '--epochs', type=int, help='Number of training epochs')
    training_parser.add_argument('-nh', '--hidden_units', type=int, help='Number of units of BLSTM cells')
    training_parser.add_argument('-nl', '--layers', type=int, help='Number of stacked BLSTM cells')
    training_parser.add_argument('-d', '--dropout', type=float, default=1,
                                 help='Dropout rate (default: 1)')
    training_parser.add_argument('-r', '--regularization', type=float, default=0,
                                 help='Weights regularization hyperparameter (default: 0)')
    training_parser.add_argument('-mt', '--mask_threshold', type=float, default=-1,
                                 help='Threshold on estimated TBM for reconstruction ("vl2m" model only). If -1 (default) thresholding is not applied')

    # Eval audio-visual speech enhancement model
    testing_parser = subparsers.add_parser('testing', description="""Test an audio-visual speech enhancement model.""")
    testing_parser.add_argument('-data', '--data_dir', required=True, help='The base pathname of dataset')
    testing_parser.add_argument('-ts', '--test_set', required=True, help='Subdirectory with TFRecords of test set')
    testing_parser.add_argument('-e', '--exp', required=True, help='Experiment name to be evaluated')
    testing_parser.add_argument('-c', '--ckp', required=True,
                                help='Model checkpoint to be restored. The format is <n_epoch>_<n_step>')
    testing_parser.add_argument('-m', '--mode', default='fixed', choices=['fixed', 'var'],
                                help='TFRecord type: "fixed" (default) or "var"')
    testing_parser.add_argument('-vd', '--video_dim', type=int, default=136,
                                help='Size of a single video frame (default: 136)')
    testing_parser.add_argument('-ad', '--audio_dim', type=int, default=257,
                                help='Size of a single audio frame (default: 257)')
    testing_parser.add_argument('-mt', '--mask_threshold', type=float, default=-1,
                                help='Threshold on estimated TBM for reconstruction. If -1 (default) thresholding is not applied')
    testing_parser.add_argument('-ns', '--num_audio_samples', type=int,
                                help='Number of samples of audio wav if <mode> is "fixed" (otherwise it is ignored)')
    testing_parser.add_argument('-me', '--mix_eval', action='store_const', const=True, default=False, 
                                help='If it is set mixed-speech wavs are evaluated')
    testing_parser.add_argument('-od', '--output_dir', default='',
                                help='Directory where are saved enhanced, mixed and target wavs. If empty string no wavs are saved')
    testing_parser.add_argument('-md', '--mask_dir', default='',
                                help='Subdirectory <data_dir>/s<base_speaker_id>/<mask_dir> where estimated masks (spectrograms for "av_concat_spec" model')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.subparser_name == 'mixed_speech_generator':
        create_mixed_tracks_data(args.data_dir, args.base_speaker_ids, args.noisy_speaker_ids, args.audio_dir,
                                 args.dest_dir, args.num_samples, args.num_mix, args.num_mix_speakers)
    elif args.subparser_name == 'audio_preprocessing':
        save_spectrograms(args.data_dir, args.speaker_ids, args.audio_dir, args.dest_dir,
                          args.sample_rate, args.max_wav_length)
    elif args.subparser_name == 'video_preprocessing':
        save_face_landmarks(args.data_dir, args.speaker_ids, args.video_dir, args.dest_dir, args.shape_predictor, args.ext)
    elif args.subparser_name == 'show_face_landmarks':
        show_face_landmarks(args.video, fps=args.fps, predictor_params=args.shape_predictor, landmarks_file=args.face_landmarks, full_draw=args.full)
    elif args.subparser_name == 'tbm_computation':
        save_target_binary_masks(args.data_dir, args.speaker_ids, args.audio_dir, args.dest_dir, args.mask_factor,
                                 args.sample_rate, args.max_wav_length, args.num_ltass)
    elif args.subparser_name == 'tfrecords_generator':
        create_dataset(args.data_dir, args.num_speakers, args.video_dir, args.tbm_dir, args.base_audio_dir, args.mix_audio_dir, args.norm_data_dir,
                       args.dest_dir, args.mode, args.delta)
    elif args.subparser_name == 'training':
        config = Configuration(args.learning_rate, args.updating_step, args.learning_decay, args.dropout, args.batch_size,
                               args.opt, args.video_dim, args.audio_dim, args.num_audio_samples, args.epochs, args.hidden_units,
                               args.layers, args.regularization, args.mask_threshold)
        train(args.model, args.data_dir, args.train_set, args.val_set, config, args.exp, args.mode)
    elif args.subparser_name == 'testing':
        test(args.data_dir, args.test_set, args.exp, args.ckp, args.video_dim, args.audio_dim, args.mode,
             args.num_audio_samples, args.mask_threshold, args.mix_eval, args.output_dir, args.mask_dir)
    else:
        print("Bad subcommand name.")
        sys.exit(1)


if __name__ == '__main__':
    main()
