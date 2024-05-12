import React, { useEffect, useRef, useState } from "react";

function App() {
  const inputFileElemenetRef = useRef<HTMLInputElement>(null);
  const sourceImageElementRef = useRef<HTMLImageElement>(null);
  const fileReaderRef = useRef<FileReader>(new FileReader());
  const [images, setImages] = useState<string[]>([
    "https://dummyimage.com/450x300/dee2e6/6c757d.jpg",
    "https://dummyimage.com/450x300/dee2e6/6c757d.jpg",
    "https://dummyimage.com/450x300/dee2e6/6c757d.jpg",
    "https://dummyimage.com/450x300/dee2e6/6c757d.jpg",
  ]);

  useEffect(() => {
    fileReaderRef.current.onload = () => {
      sourceImageElementRef!.current!.src = fileReaderRef.current.result as string
    }
  }, [])

  function handleFileUpload() {
    inputFileElemenetRef?.current?.click()
  }

  function handleFileSelected(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files![0];
    const formData = new FormData();
    formData.append("file", file);
    fileReaderRef.current.readAsDataURL(file)
    fetch("http://localhost:5000/", {
      method: "POST",
      body: formData,
    })
      .then((resp) => resp.json())
      .then((resp) => {
        setImages(resp.images.map((image: string) => `/images/${image}`));
      });

    e.target.value = "";
  }

  return (
    <>
      <nav className="navbar navbar-expand-lg navbar-light bg-light">
        <div className="container px-4 px-lg-5">
          <a className="navbar-brand" href="#!">
            Start Bootstrap
          </a>
          <button
            className="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarSupportedContent"
            aria-controls="navbarSupportedContent"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarSupportedContent">
            <ul className="navbar-nav me-auto mb-2 mb-lg-0 ms-lg-4">
              <li className="nav-item">
                <a className="nav-link active" aria-current="page" href="#!">
                  Home
                </a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="#!">
                  About
                </a>
              </li>
              <li className="nav-item dropdown">
                <a
                  className="nav-link dropdown-toggle"
                  id="navbarDropdown"
                  href="#"
                  role="button"
                  data-bs-toggle="dropdown"
                  aria-expanded="false"
                >
                  Shop
                </a>
                <ul className="dropdown-menu" aria-labelledby="navbarDropdown">
                  <li>
                    <a className="dropdown-item" href="#!">
                      All Products
                    </a>
                  </li>
                  <li>
                    <hr className="dropdown-divider" />
                  </li>
                  <li>
                    <a className="dropdown-item" href="#!">
                      Popular Items
                    </a>
                  </li>
                  <li>
                    <a className="dropdown-item" href="#!">
                      New Arrivals
                    </a>
                  </li>
                </ul>
              </li>
            </ul>
            <form className="d-flex">
              <button className="btn btn-outline-dark" type="submit">
                <i className="bi-cart-fill me-1"></i>
                Cart
                <span className="badge bg-dark text-white ms-1 rounded-pill">
                  0
                </span>
              </button>
            </form>
          </div>
        </div>
      </nav>

      <section className="py-5">
        <div className="container px-4 px-lg-5 my-5">
          <div className="row gx-4 gx-lg-5 align-items-center">
            <div className="col-md-6">
              <img
                className="card-img-top mb-5 mb-md-0"
                src="https://dummyimage.com/600x700/dee2e6/6c757d.jpg"
                alt="..."
                ref={sourceImageElementRef}
              />
            </div>
            <div className="col-md-6">
              <h1 className="display-5 fw-bolder">Shop item template</h1>

              <p className="lead">
                Lorem ipsum dolor sit amet consectetur adipisicing elit.
                Praesentium at dolorem quidem modi. Nam sequi consequatur
                obcaecati excepturi alias magni, accusamus eius blanditiis
                delectus ipsam minima ea iste laborum vero?
              </p>
              <div className="d-flex">
                <button
                  className="btn btn-outline-dark flex-shrink-0"
                  type="button"
                  onClick={handleFileUpload}
                >
                  <i className="bi-cart-fill me-1"></i>
                  Insert Picture
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-5 bg-light">
        <div className="container px-4 px-lg-5 mt-5">
          <h2 className="fw-bolder mb-4">Related images</h2>
          <div className="row gx-4 gx-lg-5 row-cols-2 row-cols-md-3 row-cols-xl-4 justify-content-center">
            {images.map((image, index) => (
              <div className="col mb-5" key={index}>
                <div className="card h-100">
                  <img className="card-img-top" src={image} alt="..." />

                  <div className="card-body p-4">
                    <div className="text-center">
                      <h5 className="fw-bolder">...</h5>
                      ...
                    </div>
                  </div>

                  <div className="card-footer p-4 pt-0 border-top-0 bg-transparent">
                    <div className="text-center">
                      <a className="btn btn-outline-dark mt-auto" href="#">
                        ...
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      <footer className="py-5 bg-dark">
        <div className="container">
          <p className="m-0 text-center text-white">
            Copyright &copy; Your Website {new Date().getFullYear()}
          </p>
        </div>
      </footer>
      <input
        ref={inputFileElemenetRef}
        type="file"
        onChange={handleFileSelected}
        style={{
          display: "none",
        }}
      />
    </>
  );
}

export default App;
