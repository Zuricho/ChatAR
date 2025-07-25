import os, sys, datetime
from pathlib import Path
from openai import OpenAI
import base64
import dashscope
import time



def initialize_client():
    # Initialize the aliyun client
    client = OpenAI(
        api_key="API_KEY_REQUIRED",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return client


def encode_image(image_path):
    # all input images need to be encoded in base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def list_directory_with_creation_times(directory_path='.', silent=False):
    """
    Lists all files and directories in the specified path along with their creation times.
    """
    directory = Path(directory_path)
    if not silent:
        print(f"Listing creation times for entries in: {directory.resolve()}\n")

    file_name_list = []
    create_time_list = []
    
    for path in directory.iterdir():
        try:
            stat_info = path.stat()
            # Get creation time (st_birthtime if available, else st_ctime)
            creation_time = getattr(stat_info, 'st_birthtime', stat_info.st_ctime)
            dt = datetime.datetime.fromtimestamp(creation_time)
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            entry_type = 'Directory' if path.is_dir() else 'File'
            if not silent:
                print(f"{entry_type}: {path.name} - Created at {formatted_time}")
            file_name_list.append(path.name)
            create_time_list.append(formatted_time)
        except FileNotFoundError:
            if not silent:
                print(f"Skipping {path.name}: File not found.")
        except PermissionError:
            if not silent:
                print(f"Skipping {path.name}: Permission denied.")
        except Exception as e:
            if not silent:
                print(f"Error processing {path.name}: {e}")
    
    return file_name_list, create_time_list


def get_latest_file_path(input_folder_path):
    """
    Get the latest file from the input folder based on creation time.
    """
    # get file list and return the latest file path
    file_name_list, create_time_list = list_directory_with_creation_times(input_folder_path, silent=True)

    if len(file_name_list) == 0:
        return None

    latest_file_name = file_name_list[create_time_list.index(max(create_time_list))]
    latest_file_path = os.path.join(input_folder_path, latest_file_name)
    return latest_file_path



def main(input_folder_path):
    # Initialize the OpenAI client
    client = initialize_client()

    display_text_path = "./AR_js/input2web/input2web.txt"
    if not os.path.exists(display_text_path):
        with open(display_text_path, "w") as file:
            file.write("")

    persona = "Unknown"

    current_file = None

    while True:
        # get the latest file
        latest_file_path = get_latest_file_path(input_folder_path)
        
        if latest_file_path == current_file or latest_file_path is None:
            # If the latest file is the same as the current file, wait and continue
            time.sleep(3)
            continue

        file_type = latest_file_path.split(".")[-1]
        current_file = latest_file_path

        if file_type in ["jpg", "jpeg", "png", "webp"]:
            # Encode the image
            encoded_image = encode_image(latest_file_path)

            if file_type == "jpg" or file_type == "jpeg":
                image_url = f"data:image/jpeg;base64,{encoded_image}"
            elif file_type == "png":
                image_url = f"data:image/png;base64,{encoded_image}"
            elif file_type == "webp":
                image_url = f"data:image/webp;base64,{encoded_image}"
            else:
                raise ValueError("Unsupported image format. Supported formats are: jpg, jpeg, png, webp.")


            completion = client.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}, 
                            },
                            {"type": "text", "text": "What is in the center of the image? Reply briefly with no more than five words."},
                        ],
                    },
                ],
            )

            reply_message = completion.choices[0].message.content

            print(f"Reply message: {reply_message}")

            with open(display_text_path, "w") as file:
                file.write(reply_message)

            # update persona
            persona = reply_message


        elif file_type in ["mp3", "webm"]:
            messages = [
                {
                    "role": "system", 
                    "content": [{"text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    # "content": [{"audio": latest_file_path}, {"text": "What is the content of the audio?"}],
                    "content": [{"audio": latest_file_path}, {"text": f"Pretend you are {persona}, Reply to the audio message briefly with no more than ten words."}],
                }
            ]

            completion = dashscope.MultiModalConversation.call(
                model="qwen-audio-turbo-latest", 
                messages=messages,
                result_format="message"
                )

            reply_message = completion.output.choices[0].message.content[0]["text"]

            print(reply_message)

            with open(display_text_path, "w") as file:
                file.write(reply_message)


        else:
            raise ValueError("Unsupported file type. Supported types are: jpg, jpeg, png, webp, mp3, webm.")

        # wait
        time.sleep(3)




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_image_info.py <image_path>")
    else:
        main(sys.argv[1])

