/**
 * CropAid - Frontend JavaScript functionality
 * This script handles the image upload functionality, preview, form validation, and API key visibility toggle
 */

// Wait for the DOM to be fully loaded before executing any code
document.addEventListener('DOMContentLoaded', function () {
    // Get references to important DOM elements we'll need
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewImage = document.getElementById('preview-image');
    const uploadPrompt = document.getElementById('upload-prompt');
    const submitBtn = document.getElementById('submit-btn');
    const uploadForm = document.getElementById('upload-form');
    const toggleApiKeyBtn = document.getElementById('toggle-api-key');
    const apiKeyInput = document.getElementById('api_key');

    // Initialize API key visibility toggle
    if (toggleApiKeyBtn && apiKeyInput) {
        toggleApiKeyBtn.addEventListener('click', function() {
            // Toggle between password and text type
            const type = apiKeyInput.getAttribute('type') === 'password' ? 'text' : 'password';
            apiKeyInput.setAttribute('type', type);
            
            // Update icon based on visibility state
            const icon = toggleApiKeyBtn.querySelector('i');
            if (type === 'text') {
                icon.classList.remove('fa-eye-slash');
                icon.classList.add('fa-eye');
            } else {
                icon.classList.remove('fa-eye');
                icon.classList.add('fa-eye-slash');
            }
        });
    }

    // Only initialize if we're on the upload page (these elements exist)
    if (uploadArea && fileInput) {
        // Add click event to the upload area to trigger file input click
        uploadArea.addEventListener('click', function () {
            fileInput.click();
        });

        // Handle drag-over event for better UX during drag and drop
        uploadArea.addEventListener('dragover', function (e) {
            e.preventDefault(); // Prevent default behavior
            e.stopPropagation();
            uploadArea.classList.add('border-primary'); // Visual feedback
        });

        // Handle drag-leave event to reset visual feedback
        uploadArea.addEventListener('dragleave', function (e) {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.remove('border-primary');
        });

        // Handle file drop event
        uploadArea.addEventListener('drop', function (e) {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.remove('border-primary');
            
            // Get the dropped files
            if (e.dataTransfer.files.length) {
                // Update file input with the dropped file
                fileInput.files = e.dataTransfer.files;
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        // Handle file selection via normal file input method
        fileInput.addEventListener('change', function (e) {
            if (fileInput.files.length) {
                handleFileSelect(fileInput.files[0]);
            }
        });

        /**
         * Handles the file selection process
         * @param {File} file - The selected image file
         */
        function handleFileSelect(file) {
            // Check if the selected file is a valid image
            const validImageTypes = ['image/jpeg', 'image/png', 'image/jpg'];
            
            if (!validImageTypes.includes(file.type)) {
                // Alert user if an invalid file type is selected
                alert('Please select a valid image file (JPEG, JPG, or PNG)');
                return;
            }

            // Check file size (max 16MB)
            if (file.size > 16 * 1024 * 1024) {
                alert('File size exceeds 16MB. Please select a smaller image.');
                return;
            }

            // Create a preview of the selected image
            const reader = new FileReader();
            
            // Set up FileReader callback for when file loading completes
            reader.onload = function (e) {
                // Show the preview and hide upload prompt
                previewImage.src = e.target.result;
                previewImage.classList.remove('d-none');
                uploadPrompt.style.display = 'none';
                
                // Enable the submit button now that we have a valid image
                submitBtn.disabled = false;
            };
            
            // Start reading the file as a data URL
            reader.readAsDataURL(file);
        }

        // Add form validation before submission
        if (uploadForm) {
            uploadForm.addEventListener('submit', function (e) {
                // If no file is selected, prevent form submission
                if (!fileInput.files.length) {
                    e.preventDefault();
                    alert('Please select an image to analyze.');
                    return false;
                }
                
                // Show loading state for better UX
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
                return true;
            });
        }
    }
});