declare module "react-image-zoom" {
  import { ComponentType } from "react";

  interface ZoomProps {
    img: string;
    zoomScale: number;
    width: number;
    height: number;
  }

  const Zoom: ComponentType<ZoomProps>;
  export default Zoom;
}
