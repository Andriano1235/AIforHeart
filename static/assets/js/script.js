// Hamburger Menu
const menuToggle = document.querySelector('.menu-toggle input');
const nav = document.querySelector('nav .navbar');

menuToggle.addEventListener('click', function(){
    nav.classList.toggle('slide');
})


// Navbar After Scrolling
    window.addEventListener('scroll', function(){
        let navbar = document.querySelector('nav')
        let windowPosition = window.scrollY > 0;
        navbar.classList.toggle('scrolling-active', windowPosition);
        })


// Form Input Gambar Deteksi
function readURL(input) {
    if (input.files && input.files[0]) {
  
      var reader = new FileReader();
  
      reader.onload = function(e) {
        $('.image-upload-wrap').hide();
  
        $('.file-upload-image').attr('src', e.target.result);
        $('.file-upload-content').show();
  
        $('.image-title').html(input.files[0].name);
      };
  
      reader.readAsDataURL(input.files[0]);
  
    } else {
      removeUpload();
    }
  }
  
  function removeUpload() {
    $('.file-upload-input').replaceWith($('.file-upload-input').clone());
    $('.file-upload-content').hide();
    $('.image-upload-wrap').show();
  }
  $('.image-upload-wrap').bind('dragover', function () {
      $('.image-upload-wrap').addClass('image-dropping');
    });
    $('.image-upload-wrap').bind('dragleave', function () {
      $('.image-upload-wrap').removeClass('image-dropping');
  });