import React from "react";
import logo from "../../assets/logo1.png"
import avatar from "../../assets/account_avatar.png"

const NavBar = ()=>{
    return(
        <div className="w-full">
            <div
            className="px-4
            md:px-10
            py-5
            flex
            flex-row
            items-center
            transition
            duration-1000
            bg-colour 
            bg-opacity-90
            cursor-pointer"
            >
                <div className="cursor-pointer">
                    <img className="h-14 lg:h-17"  src={logo} alt='Logo'/>
                </div>

                <div className="flex flex-row ml-auto gap-7 items-center">

                    <div className="flex flex-row items-center gap-2 cursor-pointer relative">

                        <div className="w-10 h-10 lg:w-16 lg:h-16 overflow-hidden">
                            <img src={avatar} alt='Profile'/>
                        </div>

                    </div>

                </div>
            </div>
        </div>
    )
}

export default NavBar;