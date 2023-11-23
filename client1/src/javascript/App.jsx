import '../css/App.css';
import HomeScreen from './components/HomeScreen.jsx';
import NavBar from './components/NavBar.jsx';
// import Floating_objects from './floatingObjects.jsx';
import React from "react";


function App() {
  return (
    <div className="App" >
      {/* <Floating_objects /> */}
      <NavBar/>
      <HomeScreen className="z-100"/>
    </div>
  );
}

export default App;