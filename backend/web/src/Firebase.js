import firebase from 'firebase/compat/app';
import 'firebase/compat/firestore';
import 'firebase/compat/storage';
import 'firebase/auth';
const firebaseConfig = {

  apiKey: "AIzaSyAJ47Ipr7MfCp3fEHrYcyySebPoAP3kCCc",

  authDomain: "missing-person-finder-8e324.firebaseapp.com",

  databaseURL: "https://missing-person-finder-8e324-default-rtdb.firebaseio.com",

  projectId: "missing-person-finder-8e324",

  storageBucket: "missing-person-finder-8e324.appspot.com",

  messagingSenderId: "998844085131",

  appId: "1:998844085131:web:df150b398ffe5e84e1758e",

  measurementId: "G-3GJNXKWV2J"

};


  const firebaseApp = firebase.initializeApp(firebaseConfig);
 // const db = firebaseApp.firestore();
  //const auth =firebaseApp.auth();
  export default firebaseApp;
