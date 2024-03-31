from moviepy.editor import VideoFileClip


def cut_video(input_path, start_time, end_time, output_path):
    """
    Cuts a video between two timestamps and saves the output.

    :param input_path: Path to the input video.
    :param start_time: Start time in seconds or (hh:mm:ss).
    :param end_time: End time in seconds or (hh:mm:ss).
    :param output_path: Path to save the output video.
    """
    with VideoFileClip(input_path) as video:
        new_video = video.subclip(start_time, end_time)
        new_video.write_videofile(output_path, codec='libx264')


# Example usage
input_video_path = "D:\chrome_downloads\ANIMAL (OFFICIAL TEASER)_ Ranbir Kapoor _Rashmika M, Anil K, Bobby D _Sandeep Reddy Vanga _Bhushan K.mp4"
output_video_path = "D:\chrome_downloads\cut_video.mp4"
start_time = '00:00:58'  # Start time in hh:mm:ss
end_time = '00:01:02'  # End time in hh:mm:ss

cut_video(input_video_path, start_time, end_time, output_video_path)
