document.addEventListener('DOMContentLoaded', () => {
    const chatbotContainer = document.getElementById('chatbot-container');
    const closeChatbotBtn = document.getElementById('close-chatbot');
    const userInput = document.getElementById('user-input');
    const sendMessageBtn = document.getElementById('send-message');
    const chatMessages = document.getElementById('chat-messages');

    // Close chatbot functionality
    closeChatbotBtn.addEventListener('click', () => {
        chatbotContainer.style.display = 'none';
    });

    // Send message functionality
    sendMessageBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        // Add user message to chat
        addMessageToChatbox('user', message);

        // Send message to backend and stream response
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: message })
        })
        .then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            function processStream({ done, value }) {
                if (done) {
                    return;
                }

                const chunk = decoder.decode(value, { stream: true });
                addMessageToChatbox('bot', chunk, true);

                return reader.read().then(processStream);
            }

            return reader.read().then(processStream);
        })
        .catch(error => {
            console.error('Error:', error);
            addMessageToChatbox('bot', 'Sorry, something went wrong.');
        });

        userInput.value = '';
    }

    function addMessageToChatbox(sender, message, streaming = false) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        
        if (streaming) {
            messageElement.textContent += message;
        } else {
            messageElement.textContent = message;
        }

        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});