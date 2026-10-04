document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generate-btn');
    const promptInput = document.getElementById('prompt');
    const loadingState = document.getElementById('loading-state');
    const resultsSection = document.getElementById('results-section');
    const imageContainer = document.getElementById('image-container');
    const videoContainer = document.getElementById('video-container');
    const enhancedPromptDisplay = document.getElementById('enhanced-prompt-display');

    generateBtn.addEventListener('click', async () => {
        const promptText = promptInput.value.trim();
        if (!promptText) {
            alert('Please enter a prompt first.');
            return;
        }

        // UI Updates for loading
        generateBtn.disabled = true;
        generateBtn.style.opacity = '0.5';
        resultsSection.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            // Send request to our FastAPI backend
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt: promptText })
            });

            if (!response.ok) {
                throw new Error('Server error occurred during generation.');
            }

            const data = await response.json();

            // Clear previous results
            imageContainer.innerHTML = '';
            videoContainer.innerHTML = '';
            enhancedPromptDisplay.innerText = data.enhanced_prompt;

            // Inject Image
            if (data.image_url) {
                const img = document.createElement('img');
                img.src = data.image_url + "?t=" + new Date().getTime(); // Prevent caching
                imageContainer.appendChild(img);
            } else {
                imageContainer.innerHTML = '<p style="color:var(--text-muted)">Image generation failed.</p>';
            }

            // Inject Video
            if (data.video_url) {
                const video = document.createElement('video');
                video.src = data.video_url + "?t=" + new Date().getTime(); // Prevent caching
                video.controls = true;
                video.autoplay = true;
                video.loop = true;
                videoContainer.appendChild(video);
            } else {
                videoContainer.innerHTML = `<p style="color:var(--text-muted)">${data.message}</p>`;
            }

            // Show results
            loadingState.classList.add('hidden');
            resultsSection.classList.remove('hidden');

            // Scroll to results smoothly
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            console.error(error);
            alert('An error occurred: ' + error.message);
            loadingState.classList.add('hidden');
        } finally {
            generateBtn.disabled = false;
            generateBtn.style.opacity = '1';
        }
    });
});
