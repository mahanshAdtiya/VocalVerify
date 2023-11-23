import React, { useEffect, useState } from "react";
import { styles } from "../styles"
import logo from "../../assets/logo1.png";

const Navbar = () => {
  const [scrolled, setScrolled] = useState(false);
  const [prevScrollPos, setPrevScrollPos] = useState(0)
  const [visible, setVisible] = useState(true)  
  useEffect(() => {

    const handleScroll = () => {
        const currentScrollPos = window.pageYOffset

        if ((prevScrollPos > currentScrollPos) !== visible)
        {
            setVisible(prevScrollPos > currentScrollPos)
        }

        setPrevScrollPos(currentScrollPos)
        
    }

    window.addEventListener('scroll', handleScroll)

    return () => window.removeEventListener('scroll', handleScroll)

}, [prevScrollPos, visible])

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      if (scrollTop > 100) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener("scroll", handleScroll);

    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav
      className={`${ styles.paddingX } w-full flex items-center py-5 transition-all fixed top-0 z-20 ${visible ? "translate-y-0": "-translate-y-28"} ${ scrolled ? "bg-black bg-opacity-50 " : "bg-transparent"}`} >
      <div className='w-full flex justify-between items-center max-w-7xl mx-auto '>
        <div className="flex items-center gap-2">

          <img src={logo} alt='logo' className='w-20 h-20 object-contain' />
          <p className='text-white text-5xl font-light cursor-pointer flex font-montserrat' style={{ fontSize: '18px' }}>
            VocalVerify &nbsp;
            <span className='sm:block hidden'> | DeepFake Test</span>
          </p>

        </div>
      </div>
    </nav>
  );
};

export default Navbar;