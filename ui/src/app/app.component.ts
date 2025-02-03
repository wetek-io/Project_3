import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'ui';

  reference_images = [
    'https://ms-cdn2.maggiesottero.com/143357/High/Rebecca-Ingram-Adeline-Fit-and-Flare-Wedding-Dress-25RK278A01-PROMO1-IV.jpg',
    'https://ms-cdn2.maggiesottero.com/143359/High/Rebecca-Ingram-Adeline-Fit-and-Flare-Wedding-Dress-25RK278A01-PROMO2-IV.jpg',
    'https://ms-cdn2.maggiesottero.com/143365/High/Rebecca-Ingram-Adeline-Sheath-Wedding-Dress-25RK278A01-Alt50-IV.jpg',
    'https://ms-cdn2.maggiesottero.com/143373/High/Rebecca-Ingram-Adeline-Sheath-Wedding-Dress-25RK278A01-Alt54-IV.jpg',
    'https://ms-cdn2.maggiesottero.com/143371/High/Rebecca-Ingram-Adeline-Sheath-Wedding-Dress-25RK278A01-Alt53-IV.jpg',
  ]

  constructor(){}
}
