# ============================================================
# EXPANDER 4: CLINICAL CASE (WITH SAFE FILE HANDLING)
# ============================================================
with st.expander("📋 Clinical Case: Benign Polyp Responded to Dostarlimab", expanded=False):
    st.markdown("""
    ### A Surprising Validation of the STEV Model
    
    **Clinical history:**
    
    - Patient with Lynch syndrome undergoing dostarlimab immunotherapy
    - Prior **MLH1-deficient malignant tumor** (TMB = 55) showed shrinkage **faster than the STEV model's population mean**
    - A **flat, benign polyp** (approximately 5-6 mm) in the descending colon (about 30 cm from anal orifice)
    - **Could not be removed** during two colonoscopies - the second attempted by a specialist using **Endoscopic Submucosal Dissection (ESD)**, which failed due to the polyp's flat, fibrotic morphology
    
    **Outcome:**
    
    - During dostarlimab treatment, the benign polyp **shrank progressively**
    - Shrinkage was **slower than the STEV model's population mean** (unlike the faster-than-mean MLH1 tumor)
    - Both trajectories - the fast MLH1 tumor and the slower benign polyp - fell **within the model's 90% credible interval**
    - Third colonoscopy successfully removed the polyp without complications
    - Post-removal pathology confirmed **benign** histology
    """)
    
    st.markdown("### STEV Model Projection vs. Actual Polyp Measurements")
    
    # Safe file handling with try/except
    try:
        if os.path.exists("benign_polyp_STEV.png"):
            from PIL import Image
            # Try to open and verify the image
            img = Image.open("benign_polyp_STEV.png")
            img.verify()  # Verify it's a valid image
            st.image("benign_polyp_STEV.png", caption="STEV model projection (blue line with 90% credible band) vs. actual benign polyp measurements (red circles). The measured dimensions fall within the predicted credible interval, validating the model's applicability to benign MMR-deficient lesions.", use_container_width=True)
        else:
            st.warning("⚠️ Benign polyp projection plot not found. Please upload the file 'benign_polyp_STEV.png' to the repository.")
            st.info("The plot shows: STEV model projection vs. actual benign polyp measurements. The measured dimensions fall within the predicted credible interval.")
    except Exception as e:
        st.warning(f"⚠️ Could not display the image. The file may be corrupted or in an unsupported format.")
        st.info("Please re-upload 'benign_polyp_STEV.png' as a valid PNG file.")
    
    st.markdown("""
    **Why this matters:**
    
    1. **Model validation** - The STEV model's credible interval captured **both** an exceptionally fast malignant tumor and a slower benign polyp
    
    2. **Biological insight** - Response speed varies continuously:
       - MLH1, high TMB: faster than mean
       - Benign polyps: slower than mean
       - Both still within the model's stochastic range
    
    3. **Practical guidance** - For flat, unresectable polyps where ESD fails, a trial of immunotherapy may enable subsequent removal - but response may be **slower than the model's mean**
    
    4. **Future directions** - This suggests possible **chemoprevention** applications of immunotherapy in Lynch syndrome
    """)
