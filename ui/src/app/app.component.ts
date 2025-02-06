import { CommonModule } from '@angular/common';
import { HttpClient, provideHttpClient, withFetch } from '@angular/common/http';
import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule],

  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  title: string = 'ui';
  hasSelectedImage: boolean = false;
  hasUploadedImage: boolean = false;
  hasFinishedSelections: boolean = false;
  selectedImage: string = '';
  uploadedImage: string | ArrayBuffer | null = null;

  reference_images = [
    'https://ms-cdn2.maggiesottero.com/143357/High/Rebecca-Ingram-Adeline-Fit-and-Flare-Wedding-Dress-25RK278A01-PROMO1-IV.jpg',
    'https://ms-cdn2.maggiesottero.com/143359/High/Rebecca-Ingram-Adeline-Fit-and-Flare-Wedding-Dress-25RK278A01-PROMO2-IV.jpg',
    'https://ms-cdn2.maggiesottero.com/143365/High/Rebecca-Ingram-Adeline-Sheath-Wedding-Dress-25RK278A01-Alt50-IV.jpg',
    'https://ms-cdn2.maggiesottero.com/143373/High/Rebecca-Ingram-Adeline-Sheath-Wedding-Dress-25RK278A01-Alt54-IV.jpg',
    'https://ms-cdn2.maggiesottero.com/143371/High/Rebecca-Ingram-Adeline-Sheath-Wedding-Dress-25RK278A01-Alt53-IV.jpg',
  ];

  constructor(private http: HttpClient) {}

  selectImage(image: any) {
    this.hasSelectedImage = true;
    this.selectedImage = image;
    console.log(image);
  }

  reselectImage() {
    this.hasSelectedImage = !this.hasSelectedImage;
  }

  uploadImage(event: Event) {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) {
      this.hasFinishedSelections = true;
      const reader = new FileReader();
      reader.onload = () => {
        this.uploadedImage = reader.result;
      };
      reader.readAsDataURL(file);
    }
  }

  generateImage() {
    const api = 'http://127.0.0.1:8000/try-on/';
    const formData = new FormData();

    formData.append('reference_image', this.selectedImage); // Pass URL directly
    if (this.uploadedImage) {
      const userBlob = this.dataURLtoBlob(this.uploadedImage as string);
      formData.append('user_image', userBlob, 'user_image.jpg');
    }

    this.http.post(api, formData).subscribe({
      next: (response) => console.log('API Response:', response),
      error: (err) => console.error('Error:', err),
    });
  }

  // Display the processed image
  displayProcessedImage(imageUrl: string) {
    const imgElement = document.getElementById(
      'processedImage'
    ) as HTMLImageElement;
    imgElement.src = imageUrl;
  }

  // Helper function to convert base64 to Blob
  dataURLtoBlob(dataURL: string): Blob {
    const byteString = atob(dataURL.split(',')[1]);
    const mimeString = dataURL.split(',')[0].split(':')[1].split(';')[0];
    const arrayBuffer = new ArrayBuffer(byteString.length);
    const uintArray = new Uint8Array(arrayBuffer);

    for (let i = 0; i < byteString.length; i++) {
      uintArray[i] = byteString.charCodeAt(i);
    }

    return new Blob([arrayBuffer], { type: mimeString });
  }
}
