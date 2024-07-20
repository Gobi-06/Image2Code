import os
import streamlit as st
from Pix2Code_CodeGenerator import generate_gui_file,generate_html_file
from PIL import Image
from matplotlib import pyplot as plt 
import time
st.title("Image to Code Generator")

fig = plt.figure()

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    with open(os.path.join(os.getcwd(),"uploaded_files",uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.join(os.getcwd(),"uploaded_files",uploaded_file.name)


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Generate Code")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                 # Save the file locally
                image_file_path = save_uploaded_file(file_uploaded)
                st.write(f"File saved to: {image_file_path}")
                
                plt.imshow(image)
                plt.axis("off")

                with st.spinner('Generating GUI...'):
                    gui_output_path, filename = generate_gui_file(image_file_path)
                    st.info('GUI File generated successfully')
        
                with st.spinner('Generating HTML...'):
                    html_file_path = generate_html_file(gui_output_path, filename)
                    st.info('HTML file generated successfully')

                
                if gui_output_path!="":
                    gui_filepath = "{}/{}.gui".format(gui_output_path,filename)
                    print("GUI Filepath ",gui_filepath)
                    with open(gui_filepath, "r", encoding="utf-8") as f:
                        gui_content = f.read()
                        
                    st.code(gui_content, language='html')

                if html_file_path!="":
                    print("HTML filepath ",html_file_path)
                    # Read the file content and display it
                    with open(html_file_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    
                    # Display the HTML file content
                    st.components.v1.html(html_content, height=600,width=1200)
                   
                time.sleep(1)
                st.success('Code Generated Successfully')
             

if __name__=='__main__':
    main()