import streamlit as st
import sys
import threading
from PIL import Image as PILImage
from agno.media import Image
from workflow import medical_intake_workflow

# Streamlit config
st.set_page_config(page_title="Medical Intake AI", page_icon="💊", layout="centered")

st.title("Medical Document Intake Workflow")
st.markdown("Upload a medical document (Insurance Card, Lab Report, etc.) to extract and normalize the data.")

# Set up the uploader
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image_display = PILImage.open(uploaded_file)
    st.image(image_display, caption="Uploaded Document", use_container_width=True)

    if st.button("Process Document", type="primary"):
        with st.status("Processing document using Agents...", expanded=True) as status:
            log_container = st.empty()
            
            # Simple stdout/stderr capture without thread:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            class UIUpdater:
                def __init__(self):
                    self.text = ""
                def write(self, text):
                    self.text += text
                    log_container.code(self.text, language="text")
                def flush(self):
                    pass
            
            updater = UIUpdater()
            sys.stdout = updater
            sys.stderr = updater
            
            success = True
            try:
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()
                image = Image(content=image_bytes)
                
                res = medical_intake_workflow.run(
                    message="Process this medical document.",
                    additional_data={"image": image}
                )
                
                # Agno Workflow catches Exceptions internally and returns them in the response.
                if hasattr(res, "steps_with_errors") and res.steps_with_errors:
                    raise Exception(f"Steps failed: {res.steps_with_errors}")
                
                status.update(label="Document Processed Successfully!", state="complete", expanded=False)
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(error_trace) # This will now go to our UIUpdater!
                
                status.update(label="Document Processing Failed", state="error", expanded=True)
                
                # Determine if it's the 503 unavailability
                if "503" in str(e) or "503" in error_trace or "UNAVAILABLE" in error_trace or "high demand" in updater.text:
                    st.error("⚠️ **Model Unavailable**: The AI model is currently experiencing high demand. Please try again later.")
                elif "'str' object has no attribute" in updater.text or "Steps failed" in str(e):
                    st.error("⚠️ **Workflow Error**: One of the processing steps failed. Check the logs above for details.")
                else:
                    st.error(f"⚠️ **Unexpected Error**: An error occurred. Check the logs.")
                    
                success = False
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            if success:
                st.success("Workflow completed successfully!")
