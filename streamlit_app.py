import streamlit as st
import cv2
import numpy as np
import imageio
from streamlit_image_coordinates import streamlit_image_coordinates
import os
import io
from PIL import Image, ImageFile
from PIL import ImageFilter
import base64
import math

#ImageFile.LOAD_TRUNCATED_IMAGES = True

def split_image(img):
    width, height = img.size

    if width % 3 != 0: #if width is not divisible by 3
        new_width = width - (width % 3)
        img = img.resize((new_width, height))
        width = new_width

    split_width = width // 3

    img1 = img.crop((0, 0, split_width, height))
    img2 = img.crop((split_width, 0, split_width * 2, height))
    img3 = img.crop((split_width * 2, 0, width, height))

    return img1, img2, img3

def scale_image(img, scale):
    width, height = img.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    return img.resize((new_width, new_height))

def smooth_image(img, sigma):
    blured_img = img.filter(ImageFilter.GaussianBlur(sigma))
    return blured_img

def rote_image(img, angle):
    # remove black border
    img = img.convert("RGBA")
    img = img.rotate(angle, expand=True)
    img = img.convert("RGB")
    return img

def fill_background_color(img, color_hex):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    color_hex = color_hex.lstrip('#')
    r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
    color = (r, g, b)

    img = img.convert("RGBA")
    width, height = img.size
    background = Image.new("RGBA", (width, height), color + (255,))
    composite = Image.alpha_composite(background, img)

    return composite.convert("RGB")

def fill_background_img2(img, img2):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if isinstance(img2, np.ndarray):
        img2 = Image.fromarray(img2)

    # Ensure both images are in RGBA mode
    img = img.convert("RGBA")
    img2 = img2.convert("RGBA")

    # Create a blank transparent canvas
    canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))

    composite = Image.alpha_composite(canvas, img2)
    composite = Image.alpha_composite(composite, img)

    return composite

def translate_image(image, x_shift, y_shift):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # Ensure the image has an alpha channel
    image = image.convert("RGBA")

    # Create a blank transparent background
    width, height = image.size
    background = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Paste the image onto the background with the shift
    background.paste(image, (int(x_shift), int(y_shift)), image)
    return background

def align_and_fill(img1, img2, img3, shift1_x, shift1_y, shift3_x, shift3_y, color):
    aligned_img1 = translate_image(img1, shift1_x, shift1_y)
    aligned_img3 = translate_image(img3, shift3_x, shift3_y)
    
    if background_option == 'Crop to Content':
        aligned_img1_c, img2, aligned_img3_c = crop_to_content(aligned_img1, img2, aligned_img3)
    elif background_option == 'Solid color':
        aligned_img1_c = fill_background_color(aligned_img1, color)
        aligned_img3_c = fill_background_color(aligned_img3, color)
    elif background_option == 'Image 2':
        aligned_img1_c = fill_background_img2(aligned_img1, img2)
        aligned_img3_c = fill_background_img2(aligned_img3, img2)
    
    return aligned_img1_c, img2, aligned_img3_c
    
def crop_to_content(img1, img2, img3):

    if isinstance(img1, np.ndarray):
        img1 = Image.fromarray(img1).convert("RGBA")
    if isinstance(img2, np.ndarray):
        img2 = Image.fromarray(img2).convert("RGBA")
    if isinstance(img3, np.ndarray):
        img3 = Image.fromarray(img3).convert("RGBA")

    alpha1 = np.array(img1)[:, :, 3]
    alpha2 = np.array(img2)[:, :, 3]
    alpha3 = np.array(img3)[:, :, 3]

    non_transparent = (alpha1 > 0) & (alpha2 > 0) & (alpha3 > 0)

    if not np.any(non_transparent):
        return img1, img2, img3

    coords = np.argwhere(non_transparent)
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)

    cropped_img1 = img1.crop((min_x, min_y, max_x + 1, max_y + 1))
    cropped_img2 = img2.crop((min_x, min_y, max_x + 1, max_y + 1))
    cropped_img3 = img3.crop((min_x, min_y, max_x + 1, max_y + 1))

    return cropped_img1, cropped_img2, cropped_img3

def create_gif(img1, img2, img3, shift1_x, shift1_y, shift3_x, shift3_y, frame_duration, color):
    aligned_img1, img2, aligned_img3 = align_and_fill(img1, img2, img3, shift1_x, shift1_y, shift3_x, shift3_y, color)

    frames = [aligned_img1, img2, aligned_img3, img2]
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    
    gif_bytes = io.BytesIO()
    pil_frames[0].save(
        gif_bytes,
        format='GIF',
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration,
        loop=0
    )
    gif_bytes.seek(0)
    return gif_bytes

def create_mp4_opencv(img1, img2, img3, shift1_x, shift1_y, shift3_x, shift3_y, frame_duration, duration, color):
    aligned_img1, img2, aligned_img3 = align_and_fill(img1, img2, img3, shift1_x, shift1_y, shift3_x, shift3_y, color)

    frames = [aligned_img1, img2, aligned_img3, img2]
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    
    pil_frames = [frame.convert("RGB") if frame.mode == "RGBA" else frame for frame in pil_frames]
    
    total_frames = int(duration / (frame_duration / 1000))
    frame_list = []

    for _ in range(total_frames // len(pil_frames)):
        frame_list.extend(pil_frames)

    mp4_bytes = io.BytesIO()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 Codec
    width, height = pil_frames[0].size
    fps = 1000 / frame_duration

    temp_filename = "temp_video.mp4"
    video_writer = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))

    for frame in frame_list:
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()

    with open(temp_filename, "rb") as f:
        mp4_bytes.write(f.read())

    os.remove(temp_filename)

    mp4_bytes.seek(0)

    return mp4_bytes
    

st.set_page_config(layout="wide")
st.image("logo.png", width=100)
st.title('Wiggle Image Generator')
st.markdown("""
The **Wiggle Image Creator** generates dynamic wiggle images from a single source file, with output formats available as **GIF** or **MP4**.

### How It Works
The program assumes by default that the source file contains **three equally sized sub-images** that are **evenly distributed**. To create a realistic 3D wiggle effect, a **focus point** is defined for each sub-image, aligning them accordingly. **Image 2** serves as the **reference point**. The selection of focus points can be made either **semi-automatically** or **manually**.

---

### Customizable Settings

- **Frame Duration and Video Length (MP4 only)**: Adjust the speed and total duration of the video to suit your needs.  

- **Background Options**: Since only Image 1 and Image 3 are aligned with Image 2, **empty areas** may appear. These gaps can be filled with either a **solid color** or with **Image 2** itself.  

- **Focus Point Selection**:  
  - **Semi-Automatic**: The focus point is determined based on Image 2, with the shift of points interpolated using a shift profile.  
  - **Manual**: The focus point can be set individually for each frame. **High precision** is essential here, as better focus point matching leads to a more convincing wiggle effect.  

- **Crop to Content**: The final wiggle image can be cropped to display only the area where **all three sub-images overlap**. This ensures a **clean and professional appearance** without unwanted borders.  

---
""")
st.header('Upload Image')
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file)
    
    st.header('Settings')
    on_vertical = st.toggle("Vertical Image", False)

    # Scale and blur
    scale_percent = st.slider('Scale (%)', 0, 100, 100)
    blur = st.slider('Blur (Sigma)', 0, 10, 0)
    
    # Choose background option
    background_options = ['Crop to Content', 'Image 2', 'Solid color']
    background_option = st.selectbox('Background', background_options)
    if background_option == 'Solid color':
        color = st.color_picker('Background color')
    else:
        color = None

    # Choose output type
    output_options = ['GIF', 'MP4']
    output_option = st.selectbox('Output type', output_options)
    # Durations
    frame_duration = int(st.slider('Frame duration (ms)', 0, 50, 10))
    if output_option == 'MP4':
        duration = int(st.slider('Duration (s)', 0, 10, 4))

    # Choose focus point selection
    focus_point_options = ['Manual','Linear transformation']
    focus_point_option = st.selectbox('Focus point selection', focus_point_options)
    if focus_point_option == 'Linear transformation':
        st.markdown('**Linear transformation parameters**')
        st.latex(r'x = a_x \cdot x_2 + t_x')
        st.latex(r'y = a_y \cdot y_2 + t_y')
        st.markdown('Transformations for Image 1:')
        a21_x = st.number_input('a_x', value=1.0075)
        t21_x = st.number_input('t_x', value=-219.9637)
        a21_y = st.number_input('a_y', value=1.0025)
        t21_y = st.number_input('t_y', value=-50.8040)
        st.markdown('Transformations for Image 2:')
        a23_x = st.number_input('a_x', value=0.9977)
        t23_x = st.number_input('t_x', value=-366.5745)
        a23_y = st.number_input('a_y', value=0.9847)
        t23_y = st.number_input('t_y', value=19.4259)

    if on_vertical:
        img = rote_image(img, 90)

    img1, img2, img3 = split_image(img)
    img1 = scale_image(img1, scale_percent / 100)
    img2 = scale_image(img2, scale_percent / 100)
    img3 = scale_image(img3, scale_percent / 100)

    if blur > 0:
        img1 = smooth_image(img1, blur)
        img2 = smooth_image(img2, blur)
        img3 = smooth_image(img3, blur)

    if on_vertical:
        img1 = rote_image(img1, -90)
        img2 = rote_image(img2, -90)
        img3 = rote_image(img3, -90)

    orig_width = img1.size[0]

    #left, middle, right = st.columns(3)

    if st.button("Set settings", use_container_width=True):
        st.header('Focus Point Selection')
        st.write('Click on the image windows to select focus points for alignment:')
        
    if focus_point_option == 'Linear transformation':
            click_1, click_2, click_3 = False, False, False
            coords1_s, coords2_s, coords3_s = None, None, None

            if on_vertical:
                width = 1500
                value = streamlit_image_coordinates(
                    img2,
                    key="local2",
                    width=width,
                )
                if value is not None:
                    coords2_s = [value["x"], value["y"]]
                    st.write(f"Focus point: ({coords2_s[0]}/{coords2_s[1]})", use_container_width=True)

                    x1 = a21_x * coords2_s[0] * (orig_width//width) + t21_x
                    y1 = a21_y * coords2_s[1] * (orig_width//width) + t21_y
                    coords1_s = [x1, y1]
                    x3 = a23_x * coords2_s[0] * (orig_width//width) + t23_x
                    y3 = a23_y * coords2_s[1] * (orig_width//width) + t23_y
                    coords3_s = [x3, y3]

                    click_1, click_2, click_3 = True, True, True
                else:
                    st.write("select focus point of the center image", use_container_width=True)

            if not on_vertical:
                width = 600
                col1, col2, col3 = st.columns(3)
                with col1:
                    pass
                with col3:
                    pass
                with col2:
                    value = streamlit_image_coordinates(
                        img2,
                        key="local2",
                        width=width,
                    )
                    if value is not None:
                        coords2_s = [value["x"], value["y"]]
                        st.write(f"Focus point: ({coords2_s[0]}/{coords2_s[1]})", use_container_width=True)

                        x1 = a21_x * coords2_s[0] * (orig_width//width) + t21_x
                        y1 = a21_y * coords2_s[1] * (orig_width//width) + t21_y
                        coords1_s = [x1, y1]
                        x3 = a23_x * coords2_s[0] * (orig_width//width) + t23_x
                        y3 = a23_y * coords2_s[1] * (orig_width//width) + t23_y
                        coords3_s = [x3, y3]

                        click_1, click_2, click_3 = True, True, True
                    else:
                        st.write("select focus point of the center image", use_container_width=True)
    else:
            click_1, click_2, click_3 = False, False, False
            coords1_s, coords2_s, coords3_s = None, None, None
            col1, col2, col3 = st.columns(3)

            if on_vertical:
                width = 1500
                value = streamlit_image_coordinates(
                    img1,
                    key="local1",
                    width=width,
                )
                if value is not None:
                    coords1_s = [value["x"], value["y"]]
                    click_1 = True
                    st.write(f"Focus point: ({coords1_s[0]}/{coords1_s[1]})", use_container_width=True)
                else:
                    st.write("Select focus point of the left image", use_container_width=True)

                value = streamlit_image_coordinates(
                    img2,
                    key="local2",
                    width=width,
                )
                if value is not None:
                    coords2_s = [value["x"], value["y"]]
                    click_2 = True
                    st.write(f"Focus point: ({coords2_s[0]}/{coords2_s[1]})", use_container_width=True)
                else:
                    st.write("select focus point of the center image", use_container_width=True)
                
                value = streamlit_image_coordinates(
                    img3,
                    key="local3",
                    width=width,
                )
                if value is not None:
                    coords3_s = [value["x"], value["y"]]
                    click_3 = True
                    st.write(f"Focus point: ({coords3_s[0]}/{coords3_s[1]})", use_container_width=True)
                else:
                    st.write("select focus point of the right image", use_container_width=True)
            
            if not on_vertical:
                width = 600
                with col1:
                    value = streamlit_image_coordinates(
                        img1,
                        key="local1",
                        width=width,
                    )
                    if value is not None:
                        coords1_s = [value["x"], value["y"]]
                        click_1 = True
                        st.write(f"Focus point: ({coords1_s[0]}/{coords1_s[1]})", use_container_width=True)
                    else:
                        st.write("Select focus point of the left image", use_container_width=True)

                with col2:
                    value = streamlit_image_coordinates(
                        img2,
                        key="local2",
                        width=width,
                    )
                    if value is not None:
                        coords2_s = [value["x"], value["y"]]
                        click_2 = True
                        st.write(f"Focus point: ({coords2_s[0]}/{coords2_s[1]})", use_container_width=True)
                    else:
                        st.write("select focus point of the center image", use_container_width=True)

                with col3:
                    value = streamlit_image_coordinates(
                        img3,
                        key="local3",
                        width=width,
                    )
                    if value is not None:
                        coords3_s = [value["x"], value["y"]]
                        click_3 = True
                        st.write(f"Focus point: ({coords3_s[0]}/{coords3_s[1]})", use_container_width=True)
                    else:
                        st.write("select focus point of the right image", use_container_width=True)

    if click_1 and click_2 and click_3:
                
                scale = orig_width // width
                if focus_point_option == 'Linear transformation':
                    coords1 = coords1_s
                    coords2 = (coords2_s[0] * scale, coords2_s[1] * scale)
                    coords3 = coords3_s
                else:
                    coords1 = (coords1_s[0] * scale, coords1_s[1] * scale)
                    coords2 = (coords2_s[0] * scale, coords2_s[1] * scale)
                    coords3 = (coords3_s[0] * scale, coords3_s[1] * scale)

                shift1_x, shift1_y = coords2[0] - coords1[0], coords2[1] - coords1[1]
                shift3_x, shift3_y = coords2[0] - coords3[0], coords2[1] - coords3[1]

                if st.button(f'Create {output_option}', use_container_width=True):

                    if output_option == 'MP4':
                        # Create MP4
                        mp4_bytes = create_mp4_opencv(np.array(img1), np.array(img2), np.array(img3), shift1_x, shift1_y, shift3_x, shift3_y, frame_duration, duration, color)
                        download_path = 'wiggle_image.mp4'

                        # Download button
                        st.download_button(
                                        label=f"Download {output_option}",
                                        data=mp4_bytes,
                                        file_name=download_path,
                                        mime="video/mp4", use_container_width=True
                                   )
                    
                    if output_option == 'GIF':
                        # Create GIF
                        gif_bytes = create_gif(np.array(img1), np.array(img2), np.array(img3), shift1_x, shift1_y, shift3_x, shift3_y, frame_duration, color)
                        download_path = 'wiggle_image.gif'

                        # Download button
                        st.download_button(
                                        label=f"Download {output_option}",
                                        data=gif_bytes,
                                        file_name=download_path,
                                        mime="image/gif", use_container_width=True
                                    )

