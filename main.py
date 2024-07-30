import cv2, numpy
from skimage.metrics import (
   structural_similarity as ssim,
)

def between_images(img_A : str, img_B : str) -> float:
   try: 
      """
         Analisa o Índice de Similaridade Estrutural (SSIM) entre Duas imagens
         
         Args:
            - img_A, img_B: imagens (filename) com dimensões iguais

         Return:
            - Valor Float SSIM entre 0 e 1 -> [+Distinto] 0 < SSIM < 1 [+Parecido] 
      """

      score, diff = ssim(
         cv2.imread(fr'{img_A}', cv2.IMREAD_GRAYSCALE),
         cv2.imread(fr'{img_B}', cv2.IMREAD_GRAYSCALE),
         full=True,
         data_range=(255)
      ) 
      
      print(f"SSIM: {score:.2f}")

      mt = cv2.matchTemplate(
         cv2.imread(img_A, cv2.IMREAD_GRAYSCALE),
         cv2.imread(img_B, cv2.IMREAD_GRAYSCALE),
         cv2.TM_CCOEFF_NORMED
      )
      print(f"cv2.matchTemplate: {mt[0][0]:.2f}")

      cv2.imshow("diff", cv2.resize(diff, (0,0), fx=0.5, fy=0.5))
      cv2.waitKey(0)
      cv2.destroyAllWindows() 

      return mt
   except Exception as error:
      return error

num_SSIM = between_images("./img/img_1.jpg", "./img/img_2.jpg")

print(f"Validate: {num_SSIM}")
