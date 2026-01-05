// Navigation Smooth Scroll
document.querySelectorAll('.nav-links a').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href').substring(1);
        document.getElementById(targetId).scrollIntoView({
            behavior: 'smooth'
        });

        // Update active link
        document.querySelectorAll('.nav-links a').forEach(link => link.classList.remove('active'));
        this.classList.add('active');
    });
});

// Update active link on scroll
window.addEventListener('scroll', () => {
    let current = '';
    const sections = document.querySelectorAll('section');

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (pageYOffset >= (sectionTop - sectionHeight / 3)) {
            current = section.getAttribute('id');
        }
    });

    document.querySelectorAll('.nav-links a').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').includes(current)) {
            link.classList.add('active');
        }
    });
});

// Hero Slideshow
const slides = document.querySelectorAll('.slide');
let currentSlide = 0;

function nextSlide() {
    slides[currentSlide].classList.remove('active');
    currentSlide = (currentSlide + 1) % slides.length;
    slides[currentSlide].classList.add('active');
}

setInterval(nextSlide, 5000); // Change slide every 5 seconds

// Model Prediction Simulation
const form = document.getElementById('prediction-form');
const loader = document.getElementById('loader');
const placeholder = document.getElementById('placeholder-text');
const resultContent = document.getElementById('result-content');
const resultImg = document.getElementById('result-img');
const resType = document.getElementById('res-type');
const resConf = document.getElementById('res-conf');
const resQual = document.getElementById('res-qual');

const transientTypes = [
    { type: "Supernova Type Ia", image: "images/galaxy.png" },
    { type: "Variable Star (Cepheid)", image: "images/nebula.png" },
    { type: "Active Galactic Nucleus", image: "images/galaxy.png" },
    { type: "Tidal Disruption Event", image: "images/observatory.png" } // Just reusing existing ones for demo
];

form.addEventListener('submit', (e) => {
    e.preventDefault();

    // UI Loading State
    placeholder.style.display = 'none';
    resultContent.style.display = 'none';
    loader.style.display = 'block';

    // Simulate Network Request Delay
    setTimeout(() => {
        loader.style.display = 'none';

        // Random prediction simulation
        const randomResult = transientTypes[Math.floor(Math.random() * transientTypes.length)];
        const randomConf = (Math.random() * (99 - 85) + 85).toFixed(1);

        // Update Content
        resultImg.src = randomResult.image;
        resType.innerText = randomResult.type;
        resConf.innerText = randomConf + "%";

        // Simple logic for quality based on inputs (fake)
        const moon = parseFloat(document.querySelector('input[placeholder="e.g. 0.15"]').value) || 0;
        if (moon > 0.5) {
            resQual.innerText = "Poor (High Background)";
            resQual.style.color = "#ff4444";
        } else {
            resQual.innerText = "Excellent";
            resQual.style.color = "#00ff00";
        }

        resultContent.style.display = 'block';
    }, 2000);
});
