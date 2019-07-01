import { Subject } from "rxjs";

// laderStatus as Subject, It will work as both observer and obserable
export const loaderStatus = new Subject();
export const apiurl = 'https://predictstart-backend.herokuapp.com'