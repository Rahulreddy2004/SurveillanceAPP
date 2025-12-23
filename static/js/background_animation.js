document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('backgroundCanvas');
    if (!canvas) { // If canvas doesn't exist, animation won't run.
        console.log("Background canvas not found. Skipping animation.");
        return;
    }
    const ctx = canvas.getContext('2d');
    let animationFrameId;

    // Adjust canvas size to window
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    // Initial resize and add event listener
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const particles = [];
    const numParticles = 80; // Fewer particles for a cleaner look
    const maxLineDistance = 150; // Max distance for lines to form
    const particleSpeed = 0.3; // Slower, subtle movement

    class Particle {
        constructor(x, y) {
            this.x = x;
            this.y = y;
            this.vx = (Math.random() - 0.5) * particleSpeed * 2;
            this.vy = (Math.random() - 0.5) * particleSpeed * 2;
            this.radius = Math.random() * 1.5 + 0.5; // Smaller dots
            this.color = `rgba(10, 191, 188, ${Math.random() * 0.3 + 0.1})`; // Translucent teal
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            // Bounce off edges
            if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
            if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.fill();
        }
    }

    function initParticles() {
        particles.length = 0; // Clear existing particles
        for (let i = 0; i < numParticles; i++) {
            particles.push(new Particle(
                Math.random() * canvas.width,
                Math.random() * canvas.height
            ));
        }
    }

    function connectParticles() {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const p1 = particles[i];
                const p2 = particles[j];
                const distance = Math.sqrt(
                    (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
                );

                if (distance < maxLineDistance) {
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(10, 191, 188, ${0.3 - (distance / maxLineDistance) * 0.3})`; // Fading line
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear only what's needed for subtle effect
        // Optional: clear with a transparent black to leave subtle trails
        // ctx.fillStyle = 'rgba(18, 18, 30, 0.05)'; // A very subtle fade effect
        // ctx.fillRect(0, 0, canvas.width, canvas.height);

        connectParticles();
        particles.forEach(p => {
            p.update();
            p.draw();
        });

        animationFrameId = requestAnimationFrame(animate);
    }

    // Start/Stop animation functions
    function startAnimation() {
        if (!animationFrameId) {
            initParticles();
            animate();
            console.log("Background animation started.");
        }
    }

    function stopAnimation() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
            console.log("Background animation stopped.");
        }
    }

    // Start animation immediately
    startAnimation();

    // Optional: stop animation if tab is not active to save resources
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            stopAnimation();
        } else {
            startAnimation();
        }
    });
});
