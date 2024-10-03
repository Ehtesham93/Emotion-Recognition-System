import os
import cv2
import pandas as pd

def create_fer2013_csv():
    image_directory = 'C:/Users/Deepika/OneDrive/Documents/Emotion-recognition-master/emotions'  # Your images path
    data = []
    labels = []

    # Check if the main directory exists
    if not os.path.exists(image_directory):
        print(f"Error: The directory {image_directory} does not exist.")
        return

    for image_file in os.listdir(image_directory):
        # Check for .PNG files (case insensitive)
        if image_file.lower().endswith('.png'):
            image_path = os.path.join(image_directory, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
            if image is not None:
                data.append(image.flatten())
                # Extract label from the filename (remove the extension)
                label = image_file.split('.')[0]
                labels.append(label)
            else:
                print(f"Warning: Unable to read image {image_path}")
        else:
            print(f"Skipped: {image_file} (not a PNG file)")

    if not data:
        print("Warning: No images were found or read correctly. Check your directory structure.")
    else:
        print(f"Total images processed: {len(data)}")

    df = pd.DataFrame(data)
    df['Emotion'] = labels
    df.to_csv('fer2013.csv', index=False)  # Save the DataFrame to CSV
    print("CSV file created.")

if __name__ == "__main__":
    create_fer2013_csv()
