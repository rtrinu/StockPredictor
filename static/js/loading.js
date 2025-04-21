document.addEventListener('DOMContentLoaded', function() {
    fetch('/start_training')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                window.location.href = '/predictions';
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing your request.');
        });
});
