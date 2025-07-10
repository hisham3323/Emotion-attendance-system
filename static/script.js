document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const emotionText = document.getElementById('emotion-text');
    const confidenceValue = document.getElementById('confidence-value');
    const errorMessage = document.getElementById('error-message');
    const snapshotBtn = document.getElementById('snapshot-btn');
    const emotionBadge = document.getElementById('emotion-badge');

    let isProcessing = false;
    let captureInterval;

    // Emotion colors mapping
    const emotionColors = {
        'Happy': '#22c55e',
        'Sad': '#3b82f6',
        'Angry': '#ef4444',
        'Surprise': '#f59e0b',
        'Fear': '#8b5cf6',
        'Disgusted': '#ec4899',
        'Neutral': '#64748b'
    };

    // Initialize webcam
    async function initWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                } 
            });
            video.srcObject = stream;
            errorMessage.style.display = 'none';
            startCapture();
        } catch (err) {
            console.error('Error accessing webcam:', err);
            errorMessage.style.display = 'block';
        }
    }

    // Start capturing frames
    function startCapture() {
        captureInterval = setInterval(captureFrame, 2000);
    }

    // Stop capturing frames
    function stopCapture() {
        clearInterval(captureInterval);
    }

    // Capture and process frame
    async function captureFrame() {
        if (isProcessing) return;
        isProcessing = true;

        try {
            // Draw video frame to canvas
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert canvas to base64
            const imageData = canvas.toDataURL('image/jpeg', 0.8);

            // Send to backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            const data = await response.json();
            
            if (data.emotion) {
                updateEmotionDisplay(data.emotion, data.confidence);
            }
        } catch (err) {
            console.error('Error processing frame:', err);
        } finally {
            isProcessing = false;
        }
    }

    // Update emotion display
    function updateEmotionDisplay(emotion, confidence) {
        emotionText.textContent = emotion;
        confidenceValue.textContent = `${Math.round(confidence * 100)}%`;
        
        // Update badge color based on emotion
        const color = emotionColors[emotion] || emotionColors['Neutral'];
        emotionBadge.style.background = color;
    }

    // Take snapshot
    snapshotBtn.addEventListener('click', () => {
        const link = document.createElement('a');
        link.download = `emotion-snapshot-${Date.now()}.jpg`;
        link.href = canvas.toDataURL('image/jpeg');
        link.click();
    });

    // Initialize the app
    initWebcam();

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        stopCapture();
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
    });
});
