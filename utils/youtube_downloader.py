from pytube import YouTube
from argparse import ArgumentParser


def parse_args():
    """
    The function parses arguments passed in run command.

    Returns:
        arguments (Namespace) - Parsed arguments.
    """
    parser = ArgumentParser(
        description="The script downloads video from YouTube service to specific directory."
    )
    parser.add_argument("--url", help="The URL address to video.", type=str)
    parser.add_argument(
        "--output-dir", help="A path to a output directory.", type=str, default="./"
    )
    args = parser.parse_args()
    return args


def download_video(url: str, output_dir: str):
    """
    The function downloads video from the YouTube service.

    Arguments:
        url (str) - The URL address to video.
        output_dir (str) - A specific output directory.

    Returns:
        status (bool) - Returns True when downloading is successful, else False.
    """

    yt_object = YouTube(url)
    yt_object = yt_object.streams.get_highest_resolution()

    try:
        yt_object.download()
        return True
    except:
        return False


if __name__ == "__main__":
    args = parse_args()

    video_url = args.url
    output_dir = args.output_dir

    download_status = download_video(video_url, output_dir)

    if download_status:
        print("Video successfully downloaded!")
    else:
        print("An error occurred while downloading!")
